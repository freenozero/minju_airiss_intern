from library.utils.algorithm import algorithm
from library.background_manipulation import run as manipulation_run
from library.jitter import jitter
from library.split import split
from library.groundtruths import groundtruths


if __name__ == "__main__":
    # # train, val, test setting
    # setting = {"train":15999, "val":1999, "test":1999}
    setting = {"train":17999, "val":1999}
    
    highlow_setting = ["highlow", "high", "low"]
    dataset_setting = ["train", "val"]
    
    path = "D:/wp/data/manipulation_image"
    
    # 필요한 폴더, data.json 생성
    algorithm.mkDirFolder(path, highlow_setting, dataset_setting)

    # highlow에 train, val, test 나눠서 image 생성
    print("start manipulation")
    manipulation_run(setting)

    # highlow에 있는 image들을 jitter 후 저장
    print("start jitter")
    for dataset in dataset_setting:
        jit = jitter(f"{path}/highlow/{dataset}")
        jit.run()
    
    # highlow에 저장된 jitter image, data.json들을 high, low로 나눠서 저장
    print("start split")
    highlow_split = split(path)
    highlow_split.run()

    # (highlow), high, low 폴더에 있는 (train), val, test jitter image들을 groundtruth하기
    print("start groundtruths")
    for highlow in highlow_setting[1:]:
        for dataset in dataset_setting[1:]:
            ground = groundtruths(f"{path}/{highlow}/{dataset}/jitter_image", f"{path}/{highlow}/{dataset}/json/data.json", f"{path}/{highlow}/{dataset}/ground_truths")
            ground.run()