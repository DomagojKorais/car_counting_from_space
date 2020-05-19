from fastai.vision import *
from fastai.metrics import error_rate

#parse image paths from their name
def parse_path(path):
    #return false_path[23:38]
    p = re.compile('([0-9]+.[0-9]+.[0-9]+.[0-9]+).png')
    m = p.search(str(path))
    return m.group(1)

#import data
dataset = "Potsdam_ISPRS"
data_path = Path("../storage/data/")/dataset
transformed_data_folder = "all_data_transformed"

#define paths
path_train_img = Path("../storage/data/" +dataset +"/train")
path_train_anno = Path("../storage/labels/COWC_train_list_64_class.txt")
path_test_img = Path("../storage/data/" +dataset +"/test")
path_test_anno = Path("../storage/labels/COWC_test_list_64_class.txt")

fnames_train = get_image_files(path_train_img)
fnames_test = get_image_files(path_test_img)
filenames_train = pd.DataFrame(fnames_train, columns = ["path"])
filenames_train["type"] = False
filenames_test = pd.DataFrame(fnames_test, columns = ["path"])
filenames_test["type"] = True
filenames = filenames_train.append(filenames_test)


filenames['img_id'] = filenames.apply (lambda row: parse_path(row["path"]), axis=1)
annotations_train = pd.read_csv(path_train_anno, header= None, sep = " ", index_col = None, names = ["img_false_path","count"])
annotations_train = annotations_train[annotations_train['img_false_path'].str.contains(dataset)]
annotations_train['img_id'] = annotations_train.apply (lambda row: parse_path(row["img_false_path"]), axis=1)

annotations_test = pd.read_csv(path_test_anno, header= None, sep = " ", index_col = None, names = ["img_false_path","count"])
annotations_test = annotations_test[annotations_test['img_false_path'].str.contains(dataset)]
annotations_test['img_id'] = annotations_test.apply (lambda row: parse_path(row["img_false_path"]), axis=1)

annotations = annotations_train.append(annotations_test)
annotations = pd.merge(annotations, filenames)

np.random.seed(42)
planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
bs=256
data = (ImageList.from_df(df = annotations,  path = data_path, folder=transformed_data_folder, cols = ['img_id'], suffix='.png')
        #Where to find the data? -> in planet 'train' folder
        .split_from_df(col="type")
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_df(cols = ["count"],label_cls=FloatList)
        #How to label? -> use the second column of the csv file and split the tags by ' '
        .transform(planet_tfms, size=256)
        #Data augmentation? -> use tfms with a size of 128
        .databunch(bs=bs)
        .normalize(imagenet_stats))                          
        #Finally -> use the defaults for conversion to databunch
    
learn = cnn_learner(data, models.resnet34,loss_func = MSELossFlat(), metrics = mean_squared_error)
learn.to_fp16() #mixed precision training (faster and less memory intensive)
#train head of model
lr = 0.01
learn.fit_one_cycle(4,slice(lr))
lr = 0.003
learn.fit_one_cycle(10,slice(lr))
learn.save()
