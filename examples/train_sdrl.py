from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import os

from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from sdrl import datasets
import maximum_mean_discrepancy
from sdrl.models import sdrl_resnet50
from sdrl.loss import CameraProxy, MSCMemory
from sdrl.trainers import SDRLTrainer
from sdrl.evaluators import Evaluator, extract_all_features
from sdrl.utils.data import IterLoader
from sdrl.utils.data import transforms as T
from sdrl.utils.data.sampler import RandomMultipleGallerySampler
from sdrl.utils.data.preprocessor import Preprocessor
from sdrl.utils.data.dam import DAMPreprocessor, DAMBatchSampler
from sdrl.utils.logging import Logger
from sdrl.utils.faiss_rerank import compute_ranked_list, compute_jaccard_distance

best_mAP = 0


def get_data(name, data_dir):
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, use_dam=False,
                     mask_suffix="_mask.png", bg_suffix="_bg.png", return_mask=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        base_sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        base_sampler = RandomSampler(train_set) if use_dam else None

    if use_dam:
        # DAM: 16 normal + 16 composite per batch -> batch_size must be 32
        dam_batch_size = 32
        preprocessor = DAMPreprocessor(
            train_set, root=dataset.images_dir, transform=train_transformer,
            height=height, width=width, mask_suffix=mask_suffix, bg_suffix=bg_suffix,
            return_mask=return_mask
        )
        batch_sampler = DAMBatchSampler(base_sampler, batch_size=dam_batch_size, drop_last=True)
        train_loader = IterLoader(
            DataLoader(preprocessor, batch_sampler=batch_sampler, num_workers=workers,
                       pin_memory=True, collate_fn=None),
            length=iters
        )
    else:
        preprocessor = Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer)
        train_loader = IterLoader(
            DataLoader(preprocessor, batch_size=batch_size, num_workers=workers, sampler=base_sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True),
            length=iters
        )

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def compute_pseudo_labels(features, cluster, k1):
    mat_dist = compute_jaccard_distance(features, k1=k1, k2=6)
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids



def compute_semantic_consistency(features_g, features_p, k, search_option=0):
    """Compute semantic consistency score between global and local features
    
    Args:
        features_g: Global features [N, D]
        features_p: Local features [N, D, P]
        k: Number of neighbors for consistency computation
        search_option: Search option for nearest neighbor computation
        
    Returns:
        consistency_scores: Semantic consistency score [N, P]
    """
    print("Compute semantic consistency score...")
    N, D, P = features_p.size()
    score = torch.zeros(N, P, device=features_g.device)
    end = time.time()
    
    # Compute ranking list for global features
    ranked_list_g = compute_ranked_list(features_g, k=k, search_option=search_option, verbose=False)
    
    # Pre-compute k-nearest neighbors for global features
    gb_neighbors_all = torch.stack([features_g[ranked_list_g[j]] for j in range(N)])  # shape: [N, k, D]
    
    # Process in batches to reduce memory usage
    initial_batch_size = 32
    min_batch_size = 16
    
    # Process in batches to reduce memory usage
    for i in range(P):
        # Compute ranking list for local features
        ranked_list_p_i = compute_ranked_list(features_p[:, :, i], k=k, search_option=search_option, verbose=False)
        
        # Process samples in batches
        for batch_start in range(0, N, initial_batch_size):
            batch_end = min(batch_start + initial_batch_size, N)
            batch_size_actual = batch_end - batch_start
            
            # Get k-nearest neighbors for local features in current batch
            pt_neighbors_batch = torch.stack([
                features_p[:, :, i][ranked_list_p_i[j]] for j in range(batch_start, batch_end)
            ])  # shape: [batch_size, k, D]
            
            # Get k-nearest neighbors for global features in current batch
            gb_neighbors_batch = gb_neighbors_all[batch_start:batch_end]  # shape: [batch_size, k, D]
            
            # Compute distance matrices in batch
            for j in range(batch_size_actual):
                # Compute distance matrices
                gb_dist = torch.cdist(gb_neighbors_batch[j], gb_neighbors_batch[j])  # shape: [k, k]
                pt_dist = torch.cdist(pt_neighbors_batch[j], pt_neighbors_batch[j])  # shape: [k, k]
                
                # Add distance normalization
                gb_dist = gb_dist / (gb_dist.max() + 1e-8)
                pt_dist = pt_dist / (pt_dist.max() + 1e-8)
                
                # Flatten distance matrices to 1D vectors
                gb_dist_flat = gb_dist.view(-1)  # shape: [k*k]
                pt_dist_flat = pt_dist.view(-1)  # shape: [k*k]
                
                # Compute MMD score
                mmd_score = maximum_mean_discrepancy.mmd_loss(
                    gb_dist_flat.unsqueeze(0),  # shape: [1, k*k]
                    pt_dist_flat.unsqueeze(0)   # shape: [1, k*k]
                ) / (k * 2)
                
                # Add stability control
                mmd_score = torch.clamp(mmd_score, min=0.0, max=1.0)
                
                score[batch_start + j, i] = mmd_score
    
    print("semantic consistency score time cost: {}".format(time.time() - end))
    
    return score

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)


def apply_sdrl_safe_defaults(args):
    """Enable lightweight paper-aligned options only with --sdrl-paper-lite (off by default to preserve accuracy)."""
    if not getattr(args, 'sdrl_paper_lite', False):
        return args
    notes = []
    if args.use_dam and args.use_ssdm and not args.no_ssdm_mask and not args.use_mask:
        args.use_mask = True
        notes.append('use_mask=True (SSDM lightweight mask; image still uses original transform)')
    if args.ssdm_mask_alpha >= 0.99:
        args.ssdm_mask_alpha = 0.5
        notes.append('ssdm_mask_alpha=0.5 (soft mask blend to reduce accuracy drop)')
    if notes:
        print('==> SDRL paper-lite: ' + '; '.join(notes))
    return args


def main_worker(args):
    global best_mAP

    args = apply_sdrl_safe_defaults(args)

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # dataset
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                     testset=sorted(dataset.train))

    # model
    num_part = args.part
    model = sdrl_resnet50(num_parts=args.part, num_classes=3000,
                         use_ssdm=args.use_ssdm, ddim_steps=args.ddim_steps,
                         ssdm_train_refine=args.ssdm_train_refine,
                         ssdm_refine_t_max=args.ssdm_refine_t_max,
                         ssdm_mask_alpha=args.ssdm_mask_alpha)
    model.cuda()
    model = nn.DataParallel(model)

    # evaluator
    evaluator = Evaluator(model)

    # optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    score_log = torch.FloatTensor([])
    for epoch in range(args.epochs):
        features_g, features_p, _ = extract_all_features(model, cluster_loader)
        features_g = torch.cat([features_g[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        features_p = torch.cat([features_p[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

        if epoch == 0:
            cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=8)

        # assign pseudo-labels
        pseudo_labels, num_class = compute_pseudo_labels(features_g, cluster, args.k1)

        # compute semantic consistency
        score = compute_semantic_consistency(features_g, features_p, k=args.k)
        score_log = torch.cat([score_log, score.unsqueeze(0)], dim=0)

        # generate new dataset with pseudo-labels
        num_outliers = 0
        new_dataset = []

        idxs, cids, pids = [], [], []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            pid = label.item()
            if pid >= num_class:  # append data except outliers
                num_outliers += 1
            else:
                new_dataset.append((fname, pid, cid))
                idxs.append(i)
                cids.append(cid)
                pids.append(pid)

        train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size,
                                        args.workers, args.num_instances, args.iters, trainset=new_dataset,
                                        use_dam=args.use_dam, mask_suffix=args.dam_mask_suffix,
                                        bg_suffix=args.dam_bg_suffix, return_mask=args.use_mask)

        # statistics of clusters and un-clustered instances
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(epoch, num_class,
                                                                                           num_outliers))

        # reindex
        idxs, cids, pids = np.asarray(idxs), np.asarray(cids), np.asarray(pids)
        features_g = features_g[idxs, :]
        features_p = features_p[idxs, :, :]
        score = score[idxs, :]

        # compute cluster centroids and camera-aware proxies
        centroids_g, centroids_p = [], []
        cam_proxy, cam_proxy_p, cam_proxy_pids, cam_proxy_cids = [], [], [], []
        for pid in sorted(np.unique(pids)):  # loop all pids
            idxs_p = np.where(pids == pid)[0]
            centroids_g.append(features_g[idxs_p].mean(0))
            centroids_p.append(features_p[idxs_p].mean(0))

            for cid in sorted(np.unique(cids[idxs_p])):  # loop all cids for pid
                idxs_c = np.where(cids == cid)[0]
                idxs_cp = np.intersect1d(idxs_p, idxs_c)
                cam_proxy.append(features_g[idxs_cp].mean(0))
                cam_proxy_p.append(features_p[idxs_cp].mean(0))
                cam_proxy_pids.append(pid)
                cam_proxy_cids.append(cid)

        centroids_g = F.normalize(torch.stack(centroids_g), p=2, dim=1)
        model.module.classifier.weight.data[:num_class].copy_(centroids_g)
        memory = CameraProxy(centroids_g.size(1), len(cam_proxy_pids)).cuda()
        memory.proxy = F.normalize(torch.stack(cam_proxy), p=2, dim=1).cuda()
        memory.pids = torch.Tensor(cam_proxy_pids).long().cuda()
        memory.cids = torch.Tensor(cam_proxy_cids).long().cuda()

        memory_p = []
        for i in range(num_part):
            centroids_p_i = torch.stack(centroids_p)[:, :, i]
            centroids_p_i = F.normalize(centroids_p_i, p=2, dim=1)
            classifier_p_i = getattr(model.module, 'classifier' + str(i))
            classifier_p_i.weight.data[:num_class].copy_(centroids_p_i)

            memory_p_i = CameraProxy(centroids_g.size(1), len(cam_proxy_pids)).cuda()
            cam_proxy_p_i = torch.stack(cam_proxy_p)[:, :, i]
            memory_p_i.proxy = F.normalize(cam_proxy_p_i, p=2, dim=1).cuda()
            memory_p_i.pids = torch.Tensor(cam_proxy_pids).long().cuda()
            memory_p_i.cids = torch.Tensor(cam_proxy_cids).long().cuda()
            memory_p.append(memory_p_i)

        # Full-gallery memory bank for MSC instance constraint (Eq.13-16); init Ws/We from source features
        msc_memory = None
        if args.use_msc and args.msc_mem:
            msc_memory = MSCMemory(features_g.size(0), features_g.size(1),
                                   tau=args.msc_tau, momentum=args.msc_mem_momentum,
                                   k_nn=args.msc_knn).cuda()
            msc_memory.init_bank(features_g,
                                 torch.from_numpy(pids).long(),
                                 torch.from_numpy(cids).long())

        # training
        trainer = SDRLTrainer(model, score, memory, memory_p, num_class=num_class, num_part=num_part,
                                 beta=args.beta, lces_part_epoch=args.lces_part_epoch, lam_cam=args.lam_cam,
                                 use_msc=args.use_msc, lam_msc=args.lam_msc, msc_tau=args.msc_tau,
                                 lam_ssdm=args.lam_ssdm, lam_sdc=args.lam_sdc,
                                 ssdm_warmup_epochs=args.ssdm_warmup_epochs,
                                 ssdm_ramp_epochs=args.ssdm_ramp_epochs,
                                 dam_exclude_comp_loss=args.dam_exclude_comp_loss,
                                 use_mask=args.use_mask, msc_memory=msc_memory)

        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))
        lr_scheduler.step()

        # evaluation
        if ((epoch+1) % args.eval_step == 0) or (epoch == args.epochs-1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
            print('\n* Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}\n'.format(epoch, mAP, best_mAP))

    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))
    np.save(osp.join(args.logs_dir, 'scores.npy'), score_log.numpy())

    # Results (default k-reciprocal re-ranking; output matches standard eval format)
    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, rerank=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semantic-Aware Disentanglement Representation Learning (SDRL)")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/test'))

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)

    # SDRL
    parser.add_argument('--part', type=int, default=3, help="number of part")
    parser.add_argument('--k', type=int, default=20,
                        help="hyperparameter for semantic consistency score")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="weighting parameter for L_ces refinement")
    parser.add_argument('--lces-part-epoch', type=int, default=5,
                        help="starting epoch for part-level L_ces smoothing")
    parser.add_argument('--lam-cam', type=float, default=0.5,
                        help="weighting parameter of camera proxy contrastive loss")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # DAM (Disentanglement Aggregation Model)
    parser.add_argument('--use-dam', action='store_true',
                        help='Use DAM: swap person/background between pairs, 16->32 images per batch')
    parser.add_argument('--dam-mask-suffix', type=str, default='_mask.png',
                        help='Suffix for mask file (image xxx.jpg -> xxx_mask.png)')
    parser.add_argument('--dam-bg-suffix', type=str, default='_bg.png',
                        help='Suffix for precomputed background (xxx_bg.png). Run precompute script first.')
    parser.add_argument('--dam-exclude-comp-loss', action='store_true',
                        help='DAM composites used only for MSC, not main loss (default off; may reduce effective triplet batch)')
    parser.add_argument('--dam-comp-in-main-loss', action='store_true',
                        help='Explicitly keep legacy behavior (same as default; composites in main loss)')
    parser.add_argument('--use-mask', action='store_true',
                        help='SSDM uses person mask (image still uses original transform; recommend --sdrl-paper-lite)')
    parser.add_argument('--no-ssdm-mask', action='store_true',
                        help='Disable mask even with --sdrl-paper-lite')
    parser.add_argument('--sdrl-paper-lite', action='store_true',
                        help='Lightweight paper alignment: SSDM mask only; no main-loss/MSC memory changes (try for accuracy)')

    # MSC (Multi-view Similarity Consistency) loss, applied when --use-dam
    parser.add_argument('--use-msc', action='store_true',
                        help='Use MSC loss to align source vs enhanced view features (recommended with --use-dam)')
    parser.add_argument('--lam-msc', type=float, default=0.1,
                        help='Weight of MSC loss')
    parser.add_argument('--msc-tau', type=float, default=0.1,
                        help='Temperature for MSC instance constraint')
    parser.add_argument('--msc-mem', action='store_true',
                        help='MSC instance constraint via memory bank (default in-batch; experimental, may hurt accuracy)')
    parser.add_argument('--msc-in-batch', action='store_true',
                        help='Explicitly use in-batch InfoNCE (same as default)')
    parser.add_argument('--msc-mem-momentum', type=float, default=0.8,
                        help='Memory bank EMA retention ratio ε (Eq.13)')
    parser.add_argument('--msc-knn', type=int, default=32,
                        help='MSC memory same-camera k-NN size Ki,c')

    # SSDM (Semantic Spatial Diffusion Model)
    parser.add_argument('--use-ssdm', action='store_true',
                        help='Use SSDM: STN + diffusion for semantic part parameters')
    parser.add_argument('--lam-ssdm', type=float, default=0.1,
                        help='Weight of SSDM diffusion loss')
    parser.add_argument('--lam-sdc', type=float, default=0.1,
                        help='Weight of SDC decoupled contrastive loss (use with --use-ssdm)')
    parser.add_argument('--ssdm-warmup-epochs', type=int, default=0,
                        help='First N epochs: lam_ssdm=0, lam_sdc=0 to stabilize accuracy (~93%%), then add SSDM/SDC')
    parser.add_argument('--ssdm-ramp-epochs', type=int, default=0,
                        help='After warmup, linearly ramp lam_ssdm/lam_sdc during N epochs (0: no ramp)')
    parser.add_argument('--ddim-steps', type=int, default=50,
                        help='DDIM inference steps for SSDM (fewer = faster)')
    parser.add_argument('--ssdm-train-refine', action='store_true',
                        help='Train parts with low-noise single-step refined Θ (experimental; default STN theta0 for stability)')
    parser.add_argument('--ssdm-refine-t-max', type=int, default=100,
                        help='Max diffusion step t for part refinement when --ssdm-train-refine')
    parser.add_argument('--ssdm-mask-alpha', type=float, default=1.0,
                        help='SSDM mask strength a: U=x*((1-a)+a*m); 1=hard mask; --sdrl-paper-lite sets 0.5 automatically')

    # cluster
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--eps', type=float, default=0.5,
                        help="distance threshold for DBSCAN")

    main()
