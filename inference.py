import dataset
import converter
import model


import csv
import argparse
import numpy as np
from tqdm import tqdm

from torchvision.ops.boxes import nms

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--path', type=str, default='kaggle_set/stage_2_test_images', help='Path to test images')

model_cl = 'cl_png_4.pt'
model_frcnn = 'mobile_png_1.pth'

submissions = 'submission.csv'

path = parser.parse_args().path
print(path)

df = converter.return_test_df(path)

test_dataset, test_loader = dataset.create_test_loader(df)

cl_cur, frcnn_cur = model.load_models(model_cl, model_frcnn)


def print_results(model_cl, model_frcnn, generator, iou_threshold=0.15, threshold=0.8):
    cl, frcnn = model_cl, model_frcnn
    test_dataloader = generator

    with open(submissions, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['patientId', 'PredictionString']
        writer.writerow(field)
        cl.eval()
        frcnn.eval()
        with tqdm(test_dataloader, unit='batch') as tepoch:
            for data in tepoch:
                tepoch.set_description(f'Testing')
                inputs, labels = data
                inputs = inputs.detach().to('cuda')
                outputs = cl(inputs)
                res = outputs.cpu().detach().numpy()
                mask = [np.round(i) for i in res]

                for i, m in enumerate(mask):
                    if m == 0:
                        writer.writerow([(labels[i]), ""])

                    else:
                        out_box = frcnn([inputs[i]])

                        for o in out_box:
                            scores = np.array(o['scores'].cpu().detach())

                            if len(scores) < 1:
                                writer.writerow([(labels[i]), ""])
                                continue

                            box_list = []
                            boxes = np.array(o['boxes'].cpu().detach())
                            box_i = np.array(o['scores'].cpu().detach()).argmax()
                            ind = nms(o['boxes'], o['scores'], iou_threshold).detach().cpu().numpy()

                            for j, bbox in enumerate(o['boxes'][ind]):
                                if o['scores'][j] > threshold:
                                    box_list += [scores[j], int(boxes[j][0] / test_dataset.mult),
                                                 int(boxes[j][1] / test_dataset.mult),
                                                 int((boxes[j][2] - boxes[j][0]) / test_dataset.mult),
                                                 int((boxes[j][3] - boxes[j][1]) / test_dataset.mult)]

                            if len(box_list) < 1:
                                box_list = [scores[box_i], int(boxes[box_i][0] / test_dataset.mult),
                                            int(boxes[box_i][1] / test_dataset.mult),
                                            int((boxes[box_i][2] - boxes[box_i][0]) / test_dataset.mult),
                                            int((boxes[box_i][3] - boxes[box_i][1]) / test_dataset.mult)]

                            box_list = " ".join(str(element) for element in box_list)
                            writer.writerow([(labels[i]), box_list])


if __name__ == "__main__":
    print_results(cl_cur, frcnn_cur, test_loader, 0.15, 0.6)
