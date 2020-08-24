## Usage

## download data
目前只用我自己有的3000張
[`image and bounding box`](https://drive.google.com/open?id=17oOUdFZ0wZeoB_ce4yazhIF6fsFsbYiY)


## data preprocessing
我新傳的data已經改好了, 不用跑`rename.py`   
`xml2csv.py`, `yolo_split.py` 其實也已經跑了,跑完就會拿到github 上面的 `label.csv`,`yolo_train.txt`, `yolo_test.txt`

## Train model
deafult feature extractor is VGG19_bn, 1 epoch takee 55 sec for 1080ti  
要自己建個叫`models`的folder 來存model
        

    python yolo_train.py
    

## Test model
change to your own model path (line 38) in `yolo_test.py`  
result : for 0005.jpg, it will generate 0005.txt in `save_root`

    python yolo_test.py
    
## Visualize result

    python visualize.py <image.jpg> <label.txt>
 
 
 P.S 路徑可能都要改一下 `yolo_train.py` 的 `train_root` 和 `yolo_test.py` 的 `test_root`
