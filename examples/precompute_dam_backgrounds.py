from __future__ import print_function
import os
import os.path as osp
import sys
import argparse
import glob

# Add project and DAM (LAMA inpainting tools) to path
SDRL_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DAM_ROOT = osp.join(SDRL_ROOT, 'DAM')
sys.path.insert(0, DAM_ROOT)
sys.path.insert(0, SDRL_ROOT)

from inpaint_with_mask_file import inpaint_image_with_mask_file


def main():
    parser = argparse.ArgumentParser(description='Precompute DAM backgrounds (LAMA inpainting)')
    parser.add_argument('--data-dir', type=str, default='/data/txf/Market1501',
                        help='Market1501 root (or dir containing bounding_box_train)')
    parser.add_argument('--train-dir', type=str, default='bounding_box_train',
                        help='Train subdir name under data-dir')
    parser.add_argument('--lama-config', type=str,
                        default=osp.join(DAM_ROOT, 'lama/configs/prediction/default.yaml'))
    parser.add_argument('--lama-ckpt', type=str, required=True,
                        help='Path to LAMA checkpoint dir (e.g. big-lama)')
    parser.add_argument('--mask-suffix', type=str, default='_mask.png')
    parser.add_argument('--bg-suffix', type=str, default='_bg.png')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    train_path = osp.join(args.data_dir, args.train_dir)
    if not osp.isdir(train_path):
        print('Train dir not found:', train_path)
        return
    img_paths = glob.glob(osp.join(train_path, '*.jpg'))
    done = 0
    skip = 0
    fail = 0
    for i, img_path in enumerate(img_paths):
        base, ext = osp.splitext(img_path)
        mask_path = base + args.mask_suffix
        bg_path = base + args.bg_suffix
        if not osp.isfile(mask_path):
            skip += 1
            continue
        if osp.isfile(bg_path):
            done += 1
            continue
        try:
            inpaint_image_with_mask_file(
                img_path, mask_path,
                args.lama_config, args.lama_ckpt,
                device=args.device, save_path=bg_path
            )
            done += 1
        except Exception as e:
            print('Fail', img_path, e)
            fail += 1
        if (i + 1) % 100 == 0:
            print('Processed', i + 1, 'done', done, 'skip', skip, 'fail', fail)
    print('Done. Total processed:', len(img_paths), 'saved:', done, 'skip(no mask):', skip, 'fail:', fail)


if __name__ == '__main__':
    main()
