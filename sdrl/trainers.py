from __future__ import print_function, absolute_import
import time

from .evaluation_metrics import accuracy
from .loss import LCes, PartLCes, SoftTripletLoss, CrossEntropyLabelSmooth, MSCLoss, MSCMemory
from .utils.meters import AverageMeter


class SDRLTrainerBase(object):
    def __init__(self, model, score, num_class=500, num_part=6, beta=0.5, lces_part_epoch=5):
        super(SDRLTrainerBase, self).__init__()
        self.model = model
        self.score = score

        self.num_class = num_class
        self.num_part = num_part
        self.lces_part_epoch = lces_part_epoch

        self.criterion_lces = LCes(lam=beta).cuda()
        self.criterion_lces_part = PartLCes().cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()
        self.criterion_tri = SoftTripletLoss().cuda()

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        losses_lces = AverageMeter()
        losses_tri = AverageMeter()
        losses_lces_part = AverageMeter()
        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):
            data = train_dataloader.next()
            inputs, targets, ca = self._parse_data(data)

            # feedforward
            emb_g, emb_p, logits_g, logits_p = self.model(inputs)
            logits_g, logits_p = logits_g[:, :self.num_class], logits_p[:, :self.num_class, :]

            # loss
            loss_lces = self.criterion_lces(logits_g, logits_p, targets, ca)
            loss_tri = self.criterion_tri(emb_g, targets)

            loss_lces_part = 0.
            if self.num_part > 0:
                if epoch >= self.lces_part_epoch:
                    for part in range(self.num_part):
                        loss_lces_part += self.criterion_lces_part(logits_p[:, :, part], targets, ca[:, part])
                else:
                    for part in range(self.num_part):
                        loss_lces_part += self.criterion_ce(logits_p[:, :, part], targets)
                loss_lces_part /= self.num_part

            loss = loss_lces + loss_tri + loss_lces_part

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            prec, = accuracy(logits_g.data, targets.data)

            losses_lces.update(loss_lces.item())
            losses_tri.update(loss_tri.item())
            losses_lces_part.update(loss_lces_part.item())
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_ces {:.3f} ({:.3f})\t'
                      'L_ces_p {:.3f} ({:.3f})\t'
                      'L_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_lces.val, losses_lces.avg,
                              losses_lces_part.val, losses_lces_part.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, idxs = inputs
        ca = self.score[idxs]
        return imgs.cuda(), pids.cuda(), ca.cuda()


class SDRLTrainer(object):
    def __init__(self, model, score, memory, memory_p, num_class=500, num_part=6, beta=0.5, lces_part_epoch=5, lam_cam=0.5,
                 use_msc=False, lam_msc=0.1, msc_tau=0.1, lam_ssdm=0.0, lam_sdc=0.1,
                 ssdm_warmup_epochs=0, ssdm_ramp_epochs=0, dam_exclude_comp_loss=False,
                 use_mask=False, msc_memory=None):
        super(SDRLTrainer, self).__init__()
        self.model = model
        self.score = score
        self.memory = memory
        self.memory_p = memory_p

        self.num_class = num_class
        self.num_part = num_part
        self.lam_cam = lam_cam
        self.lam_msc = lam_msc
        self.lam_ssdm = lam_ssdm
        self.lam_sdc = lam_sdc
        self.ssdm_warmup_epochs = ssdm_warmup_epochs
        self.ssdm_ramp_epochs = ssdm_ramp_epochs
        self.use_msc = use_msc
        self.lces_part_epoch = lces_part_epoch
        # Scheme A: DAM composite images (last 16 in batch) used only for MSC, not main loss; default False (original behavior).
        self.dam_exclude_comp_loss = dam_exclude_comp_loss
        # use_mask: pass person mask into model.forward so SSDM U=F(x)⊗m (Eq.19/24) takes effect.
        self.use_mask = use_mask
        # msc_memory: full-gallery memory bank for MSC instance constraint (Eq.13-16). Falls back to in-batch InfoNCE when None.
        self.msc_memory = msc_memory

        self.criterion_lces = LCes(lam=beta).cuda()
        self.criterion_lces_part = PartLCes().cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()
        self.criterion_tri = SoftTripletLoss().cuda()
        # With memory bank, MSCLoss only handles distribution constraint (Eq.8-12); instance constraint handled by msc_memory
        self.criterion_msc = MSCLoss(tau=msc_tau, use_instance=(msc_memory is None)).cuda() if use_msc else None

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        losses_lces = AverageMeter()
        losses_tri = AverageMeter()
        losses_cam = AverageMeter()
        losses_lces_part = AverageMeter()
        losses_msc = AverageMeter()
        losses_ssdm = AverageMeter()
        losses_sdc = AverageMeter()

        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        loss_ssdm_b = None
        loss_sdc_b = None
        for i in range(train_iters):
            data = train_dataloader.next()
            inputs, targets, cams, ca, idxs, masks = self._parse_data(data)

            # feedforward (SSDM may return 5: loss_ssdm, or 6: loss_ssdm + loss_sdc)
            out = self.model(inputs, masks) if masks is not None else self.model(inputs)
            if len(out) == 6:
                emb_g, emb_p, logits_g, logits_p, loss_ssdm_b, loss_sdc_b = out
                if loss_ssdm_b is not None:
                    # Each GPU in DataParallel computes a scalar; average them here first
                    loss_ssdm_b = loss_ssdm_b.mean()
                    losses_ssdm.update(loss_ssdm_b.item())
                if loss_sdc_b is not None:
                    loss_sdc_b = loss_sdc_b.mean()
                    losses_sdc.update(loss_sdc_b.item())
            elif len(out) == 5:
                emb_g, emb_p, logits_g, logits_p, loss_ssdm_b = out
                loss_sdc_b = None
                if loss_ssdm_b is not None:
                    loss_ssdm_b = loss_ssdm_b.mean()
                    losses_ssdm.update(loss_ssdm_b.item())
            else:
                emb_g, emb_p, logits_g, logits_p = out
                loss_ssdm_b = None
                loss_sdc_b = None
            logits_g, logits_p = logits_g[:, :self.num_class], logits_p[:, :self.num_class, :]

            # Scheme A: for DAM batch(32), use only first 16 source images for main loss (CE/triplet/camera proxy)
            if self.dam_exclude_comp_loss and targets.size(0) >= 4 and targets.size(0) % 2 == 0:
                src = targets.size(0) // 2
                emb_g_m, emb_p_m = emb_g[:src], emb_p[:src]
                logits_g_m, logits_p_m = logits_g[:src], logits_p[:src]
                targets_m, cams_m, ca_m = targets[:src], cams[:src], ca[:src]
            else:
                emb_g_m, emb_p_m = emb_g, emb_p
                logits_g_m, logits_p_m = logits_g, logits_p
                targets_m, cams_m, ca_m = targets, cams, ca

            # loss
            loss_lces = self.criterion_lces(logits_g_m, logits_p_m, targets_m, ca_m)
            loss_tri = self.criterion_tri(emb_g_m, targets_m)
            loss_gcam = self.memory(emb_g_m, targets_m, cams_m)

            loss_lces_part = 0.
            loss_pcam = 0.
            if self.num_part > 0:
                if epoch >= self.lces_part_epoch:
                    for part in range(self.num_part):
                        loss_lces_part += self.criterion_lces_part(logits_p_m[:, :, part], targets_m, ca_m[:, part])
                        loss_pcam += self.memory_p[part](emb_p_m[:, :, part], targets_m, cams_m)
                else:
                    for part in range(self.num_part):
                        loss_lces_part += self.criterion_ce(logits_p_m[:, :, part], targets_m)
                        loss_pcam += self.memory_p[part](emb_p_m[:, :, part], targets_m, cams_m)
                loss_lces_part /= self.num_part
                loss_pcam /= self.num_part

            loss_cam = loss_pcam + loss_gcam
            loss = loss_lces + loss_lces_part + loss_tri + loss_cam * self.lam_cam
            # SSDM/SDC warm-up + ramp: disable auxiliary losses first, then linear ramp-up to avoid sudden accuracy drop
            warmup = getattr(self, 'ssdm_warmup_epochs', 0)
            ramp = getattr(self, 'ssdm_ramp_epochs', 0)
            if epoch < warmup:
                aux_scale = 0.0
            elif ramp > 0:
                aux_scale = min(1.0, float(epoch - warmup + 1) / float(ramp))
            else:
                aux_scale = 1.0
            eff_lam_ssdm = self.lam_ssdm * aux_scale
            eff_lam_sdc = self.lam_sdc * aux_scale
            if loss_ssdm_b is not None and eff_lam_ssdm > 0:
                loss = loss + loss_ssdm_b * eff_lam_ssdm
            if loss_sdc_b is not None and eff_lam_sdc > 0:
                loss = loss + loss_sdc_b * eff_lam_sdc

            # MSC: DAM batch (even size, first half source / second half enhanced)
            b_msc = targets.size(0)
            if self.criterion_msc is not None and b_msc >= 4 and b_msc % 2 == 0:
                half = b_msc // 2
                loss_msc_b = self.criterion_msc(emb_g, targets, cams)
                if self.msc_memory is not None:
                    loss_ins = self.msc_memory(emb_g[:half], idxs[:half], cams[:half], targets[:half],
                                               emb_g[half:], idxs[half:], cams[half:], targets[half:])
                    self.msc_memory.update(idxs[:half], emb_g[:half].detach(),
                                           idxs[half:], emb_g[half:].detach())
                    loss_msc_b = loss_msc_b + loss_ins
                loss = loss + loss_msc_b * self.lam_msc
                losses_msc.update(loss_msc_b.item())
            else:
                losses_msc.update(0.0)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            prec, = accuracy(logits_g.data, targets.data)

            losses_lces.update(loss_lces.item())
            losses_tri.update(loss_tri.item())
            losses_cam.update(loss_cam.item())
            losses_lces_part.update(loss_lces_part.item())
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log_str = ('Epoch: [{}][{}/{}]\t'
                           'Time {:.3f} ({:.3f})\t'
                           'L_ces {:.3f} ({:.3f})\t'
                           'L_ces_p {:.3f} ({:.3f})\t'
                           'L_tri {:.3f} ({:.3f})\t'
                           'L_cam {:.3f} ({:.3f})\t'
                           .format(epoch, i + 1, len(train_dataloader),
                                   batch_time.val, batch_time.avg,
                                   losses_lces.val, losses_lces.avg,
                                   losses_lces_part.val, losses_lces_part.avg,
                                   losses_tri.val, losses_tri.avg,
                                   losses_cam.val, losses_cam.avg))
                if self.criterion_msc is not None:
                    log_str += 'L_msc {:.3f} ({:.3f})\t'.format(losses_msc.val, losses_msc.avg)
                if self.lam_ssdm > 0:
                    log_str += 'L_ssdm {:.3f} ({:.3f})\t'.format(losses_ssdm.val, losses_ssdm.avg)
                if getattr(self, 'lam_sdc', 0) > 0:
                    log_str += 'L_sdc {:.3f} ({:.3f})\t'.format(losses_sdc.val, losses_sdc.avg)
                log_str += 'Prec {:.2%} ({:.2%})\t'.format(precisions.val, precisions.avg)
                print(log_str)

    def _parse_data(self, inputs):
        if len(inputs) == 6:
            imgs, _, pids, cids, idxs, masks = inputs
            masks = masks.cuda() if self.use_mask else None
        else:
            imgs, _, pids, cids, idxs = inputs
            masks = None
        ca = self.score[idxs]
        return imgs.cuda(), pids.cuda(), cids.cuda(), ca.cuda(), idxs, masks
