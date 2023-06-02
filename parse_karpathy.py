import os
import pickle, json

kagle_json = 'annotations/dataset_coco_from_kaggle.json'
new_json_train = 'post_processed_karpthy_coco/train.json'
new_json_test = 'post_processed_karpthy_coco/test.json'
new_json_val = 'post_processed_karpthy_coco/val.json'


def map_format_kaggle_to_clipcap():
    def extract_imgid_from_name(filename):
        return str(int(filename.split('.')[0].split('_')[-1]))

    with open(kagle_json) as f:
        kaggle_data = json.load(f)
    train_data = []
    test_data = []
    val_data = []
    splits = {'train': train_data, 'test': test_data, 'val': val_data, 'restval': train_data}
    out_names = {'train': new_json_train, 'test': new_json_test, 'val': new_json_val}
    for img in kaggle_data['images']:
        imgid = extract_imgid_from_name(img['filename'])
        for cap in img['sentences']:
            correct_format = {"image_id": int(imgid), "caption": cap['raw'], "id": int(cap['sentid'])}
            splits[img['split']].append(correct_format)

    DBG = False
    if not DBG:
        for name in out_names:
            with open(out_names[name], 'w') as f:
                json.dump(splits[name], f)

        for name in out_names:
            with open(out_names[name][:-5] + '_metrics_format.json', 'w') as f:
                annos = splits[name]
                ids = [{"id": int(a["image_id"])} for a in annos]
                final = {"images": ids, "annotations": annos}
                json.dump(final, f)

    if DBG:
        # rons annotations
        with open('annotations/train_caption_of_real_training.json') as f:
        # with open('../../train_caption.json') as f:
            cur_data = json.load(f)
        ids = [str(int(c['image_id'])) for c in cur_data]
        new_ids = [str(int(c['image_id'])) for c in train_data]
        ids.sort()  # inplace
        new_ids.sort()
        assert ids == new_ids
        print('OK')




def get_train_val_data(data_set_path):
    f'''

    :param data_set_path: dict. keys =   ['train', 'val', 'test'], values = path to pickl file
    :return: ds: dict:keys=['train', 'val', 'test'],values = dict:keys = list(dataset_name), values=dict:keys=key_frame,values:dict:keys=style,values=dataframe
    '''
    ds = {}
    for data_type in data_set_path:  # ['train', 'val', 'test']
        ds[data_type] = {}
        with open(data_set_path[data_type], 'rb') as r:
            data = pickle.load(r)
        for k in data:
            ds[data_type][k] = {}
            # ds[data_type][k]['factual'] = data[k]['factual']  #todo: check if there is need to concatenate factual from senticap and flickrstyle10k
            # ds[data_type][k]['img_path'] = data[k]['image_path']
            for style in data[k]:
                # if style == 'img_path':
                #     continue
                ds[data_type][k][style] = data[k][style]
    return ds

# def convert_ds_to_df(ds, data_dir):
#     df_train = None
#     df_val = None
#     df_test = None
#     for data_type in ds:  # ['train', 'val', 'test']
#         all_data = {'category': [], 'text': []}
#         for k in ds[data_type]:
#             for style in ds[data_type][k]:
#                 if style == 'image_path' or style == 'factual':
#                     continue
#                 all_data['category'].extend([style] * len(ds[data_type][k][style]))
#                 all_data['text'].extend(ds[data_type][k][style])
#         if data_type == 'train':
#             df_train = pd.DataFrame(all_data)
#             df_train.to_csv(os.path.join(data_dir, 'train.csv'))
#         elif data_type == 'val':
#             df_val = pd.DataFrame(all_data)
#             df_val.to_csv(os.path.join(data_dir, 'val.csv'))
#         elif data_type == 'test':
#             df_test = pd.DataFrame(all_data)
#             df_test.to_csv(os.path.join(data_dir, 'test.csv'))
#     return df_train, df_val, df_test

def extract_imgid_from_name(filename, dataset): # todo:daniela
    if dataset == 'flickrstyle10k':
        return str(int(filename.split('.')[0].split('_')[0]))
    if dataset == 'senticap':
        return str(filename)
def map_format_flickrstyle10k_to_clipcap():
    dataset = 'senticap' # 'flickrstyle10k' , 'senticap'
    if dataset == 'flickrstyle10k':
        base_path_flickrstyle10k = os.path.join(os.path.expanduser('~'), 'data/flickrstyle10k/annotations')
    elif dataset == 'senticap':
        base_path_flickrstyle10k = os.path.join(os.path.expanduser('~'), 'data/senticap/annotations')
    data_set_path = {'train': {}, 'val': {}, 'test': {}}
    for data_type in ['train', 'val', 'test']:
        data_set_path[data_type] = os.path.join(base_path_flickrstyle10k, data_type + '.pkl')
    ds = get_train_val_data(data_set_path)
    print('finish to load')
    out_path = os.path.join(os.path.expanduser('~'),'data/capdec')
    # kagle_json = 'annotations/dataset_coco_from_kaggle.json'
    # new_json_train = os.path.join(out_path,'post_processed_karpthy_coco/train.json')
    # new_json_test = os.path.join(out_path,'post_processed_karpthy_coco/test.json')
    # new_json_val = os.path.join(out_path,'post_processed_karpthy_coco/val.json')
    new_json_train = {'humor': os.path.join(out_path,'postprocessed_style_data/humor_train.json'), 'romantic': os.path.join(out_path,'postprocessed_style_data/roman_train.json')}
    new_json_test = {'humor': os.path.join(out_path,'postprocessed_style_data/humor_test.json'), 'romantic': os.path.join(out_path,'postprocessed_style_data/roman_test.json')}
    new_json_val = {'humor': os.path.join(out_path,'postprocessed_style_data/humor_val.json'), 'romantic': os.path.join(out_path,'postprocessed_style_data/roman_val.json')}
    new_json_train = {'positive': os.path.join(out_path,'postprocessed_style_data/positive_train.json'), 'negative': os.path.join(out_path,'postprocessed_style_data/negative_train.json')}
    new_json_test = {'positive': os.path.join(out_path,'postprocessed_style_data/positive_test.json'), 'negative': os.path.join(out_path,'postprocessed_style_data/negative_test.json')}
    new_json_val = {'positive': os.path.join(out_path,'postprocessed_style_data/positive_val.json'), 'negative': os.path.join(out_path,'postprocessed_style_data/negative_val.json')}
    if dataset == 'flickrstyle10k':
        styles=['humor','romantic']
    elif dataset == 'senticap':
        styles=['positive','negative']
    for style in styles:
        # with open(kagle_json) as f:
        #     kaggle_data = json.load(f)
        train_data = []
        test_data = []
        val_data = []
        splits = {'train': train_data, 'test': test_data, 'val': val_data, 'restval': train_data}
        out_names = {'train': new_json_train, 'test': new_json_test, 'val': new_json_val}
        i=0
        for split in ds:
            for im in ds[split]:
                imgid = extract_imgid_from_name(im,dataset)
                if split != 'test':
                    if len(ds[split][im][style])>0:
                        correct_format = {"image_id": int(imgid), "caption": ds[split][im][style][0], "id": i, "filename": ds[split][im]['image_path'].split('/')[-1]}
                    else:
                        continue
                else: #take all tests even they don't have caption of this style
                    correct_format = {"image_id": int(imgid), "id": i,
                                      "filename": ds[split][im]['image_path'].split('/')[-1]}
                i+=1
                splits[split].append(correct_format)

        DBG = False
        if not DBG:
            for name in out_names:
                with open(out_names[name][style], 'w') as f:
                    json.dump(splits[name], f)

            for name in out_names:
                with open(out_names[name][style][:-5] + '_metrics_format.json', 'w') as f:
                    annos = splits[name]
                    ids = [{"id": int(a["image_id"])} for a in annos]
                    final = {"images": ids, "annotations": annos}
                    json.dump(final, f)

        if DBG:
            # rons annotations
            with open('annotations/train_caption_of_real_training.json') as f:
                # with open('../../train_caption.json') as f:
                cur_data = json.load(f)
            ids = [str(int(c['image_id'])) for c in cur_data]
            new_ids = [str(int(c['image_id'])) for c in train_data]
            ids.sort()  # inplace
            new_ids.sort()
            assert ids == new_ids
            print('OK')
    print('finish!')


if __name__ == '__main__':
    # map_format_kaggle_to_clipcap()
    map_format_flickrstyle10k_to_clipcap()
