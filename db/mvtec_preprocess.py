import cv2
import os
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='path of original dataset', type=str, required=True)
    parser.add_argument('--save_path', help='path to save proprocessed dataset', type=str, required=True)
    parser.add_argument('--val_ratio', help='ratio of val set', type=float, default=0.05)
    args = parser.parse_args()

    return args


def mirror(image):
    image_m = image.copy()
    image_m = image_m[:, ::-1]

    return image_m


def flip(image):
    image_f = image.copy()
    image_f = image_f[::-1, :]

    return image_f


def rotation(image, range):
    _h, _w = image.shape[0: 2]
    center = (_w // 2, _h // 2)
    rot = random.uniform(range[0], range[1])
    image_r = image.copy()
    M = cv2.getRotationMatrix2D(center, rot, 1)
    image_r = cv2.warpAffine(image_r, M, (_w, _h), borderMode=cv2.BORDER_REPLICATE)

    return image_r

def crop(image, crop_size, margin):
    height, width = image.shape[0: 2]
    x_offset = random.randint(margin[0], width - crop_size[0] - margin[0])
    y_offset = random.randint(margin[0], height - crop_size[1] - margin[1])

    return image[y_offset: y_offset+crop_size[1], x_offset: x_offset+crop_size[0]]


if __name__ == '__main__':
    TEXTURE = ['carpet', 'grid', 'leather', 'tile', 'wood']
    OBJECT = ['bottle','cable','capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    args = parse_args()
    src_path = args.src_path
    save_path = args.save_path
    val_ratio = args.val_ratio

    new_set_path = os.path.join(save_path, 'mvtec_pre')
    if not os.path.exists(new_set_path):
        os.mkdir(new_set_path)

    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isfile(item_path):
            continue
        print('Arrange {}...'.format(item))
        if item in TEXTURE:
            IsTexture = True
        elif item in OBJECT:
            IsTexture = False
        else:
            raise Exception('Wrong type')

        # make item directory
        new_set_item_path = os.path.join(new_set_path, item)
        if not os.path.exists(new_set_item_path):
            os.mkdir(new_set_item_path)

        # arragne test set
        test_img_dir = os.path.join(item_path, 'test')
        gt_dir = os.path.join(item_path, 'ground_truth')
        save_test_dir = os.path.join(new_set_item_path, 'test')
        save_gt_dir = os.path.join(new_set_item_path, 'ground_truth')
        if not os.path.exists(save_test_dir):
            os.mkdir(save_test_dir)
        if not os.path.exists(save_gt_dir):
            os.mkdir(save_gt_dir)
        for ano in os.listdir(test_img_dir):
            save_test_ano_dir = os.path.join(save_test_dir, ano)
            save_gt_ano_dir = os.path.join(save_gt_dir, ano)
            if not os.path.exists(save_test_ano_dir):
                os.mkdir(save_test_ano_dir)
            if ano != 'good':
                if not os.path.exists(save_gt_ano_dir):
                    os.mkdir(save_gt_ano_dir)

            for img_name in os.listdir(os.path.join(test_img_dir, ano)):
                img_id = img_name.split('.')[0]
                img = cv2.imread(os.path.join(test_img_dir, ano, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if ano != 'good':
                    mask = cv2.imread(os.path.join(gt_dir, ano, '{}_mask.png'.format(img_id)))
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if IsTexture is True:
                    img = cv2.resize(img, (256, 256))
                    if ano != 'good':
                     mask = cv2.resize(mask, (256, 256))
                else:
                    img = cv2.resize(img, (128, 128))
                    if ano != 'good':
                     mask = cv2.resize(mask, (128, 128))
                cv2.imwrite(os.path.join(save_test_ano_dir, img_name), img)
                if ano != 'good':
                    cv2.imwrite(os.path.join(save_gt_ano_dir, '{}_mask.png'.format(img_id)), mask)

        # arrange train & val set
        train_img_dir = os.path.join(item_path, 'train', 'good')
        save_train_img_dir = os.path.join(new_set_item_path, 'train', 'good')
        if not os.path.exists(os.path.join(new_set_item_path, 'train')):
            os.mkdir(os.path.join(new_set_item_path, 'train'))
            os.mkdir(save_train_img_dir)

        save_val_img_dir = os.path.join(new_set_item_path, 'val')
        if not os.path.exists(save_val_img_dir):
            os.mkdir(save_val_img_dir)

        # get val list
        image_list = os.listdir(train_img_dir)
        image_num = len(image_list)
        val_num = int(image_num * val_ratio)
        val_id_list = []
        for _ in range(val_num):
            if len(val_id_list) >= val_num:
                break
            val_id = random.randint(0, image_num-1)
            if val_id not in val_id_list:
                val_id_list.append(val_id)

        # get & save images
        for i, image in enumerate(image_list):
            img = cv2.imread(os.path.join(train_img_dir, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_id = image.split('.')[0]

            if IsTexture is True:
                img = cv2.resize(img, (256, 256))
                if i in val_id_list:
                    cv2.imwrite(os.path.join(save_val_img_dir, '{}.png'.format(img_id)), img)
                else:
                    # crop from original image
                    img_ori = crop(img, [128, 128], [0, 0])
                    cv2.imwrite(os.path.join(save_train_img_dir, '{}_o.png'.format(img_id)), img_ori)

                    # crop from three rotation images
                    for k in range(0, 3):
                        img_r = rotation(img, [-20, 20])
                        img_r = crop(img_r, [128, 128], [30, 30])
                        cv2.imwrite(os.path.join(save_train_img_dir, '{}_r{:d}.png'.format(img_id, k)),img_r)

                    # crop from mirrored image
                    img_m = mirror(img)
                    img_m = crop(img_m, [128, 128], [0, 0])
                    cv2.imwrite(os.path.join(save_train_img_dir, '{}_m.png'.format(img_id)), img_m)

                    # crop from flipped image
                    img_f = flip(img)
                    img_f = crop(img_f, [128, 128], [0, 0])
                    cv2.imwrite(os.path.join(save_train_img_dir, '{}_f.png'.format(img_id)), img_f)

            else:
                img = cv2.resize(img, (128, 128))
                if i in val_id_list:
                    cv2.imwrite(os.path.join(save_val_img_dir, '{}.png'.format(img_id)), img)
                else:
                    # crop from original image
                    cv2.imwrite(os.path.join(save_train_img_dir, '{}_o.png'.format(img_id)), img)

                    # crop from three rotation images
                    for k in range(0, 3):
                        img_r = rotation(img, [90*(k+1), 90*(k+1)])
                        cv2.imwrite(os.path.join(save_train_img_dir, '{}_r{:d}.png'.format(img_id, k)),img_r)

                    # crop from mirrored image
                    img_m = mirror(img)
                    cv2.imwrite(os.path.join(save_train_img_dir, '{}_m.png'.format(img_id)), img_m)

                    # crop from flipped image
                    img_f = flip(img)
                    cv2.imwrite(os.path.join(save_train_img_dir, '{}_f.png'.format(img_id)), img_f)