import itertools
import torch
from math import sqrt
class Dataencoder:
    def __init__(self):
        scale = 300
        steps = [s / scale for s in (8, 16, 32, 64, 100, 300)]
        sizes = [s / scale for s in (30, 60, 111, 162, 213, 264, 315)]
        aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
        feature_map_sizes = (38, 19, 10, 5, 3, 1)
         
        # 38×38×4+19×19×6+10×10×6+5×5×6+3×3×4+1×1×4=8732
        num_layers = len(feature_map_sizes)
        anchor_boxes = []
        # 4 boxes would be 
        # each box in acnhor_boxes is tensor with [x, y, w, h]
        for i in range(num_layers):
            for y,x in itertools.product(range(feature_map_sizes[i]), repeat=2):
                w, h = sizes[i], sizes[i]
                anchor_boxes.append([(x+0.5)/feature_map_sizes[i], (y+0.5)/feature_map_sizes[i], w, h])
                w, h = sqrt(sizes[i]*sizes[i+1]), sqrt(sizes[i]*sizes[i+1])
                anchor_boxes.append([(x+0.5)/feature_map_sizes[i], (y+0.5)/feature_map_sizes[i], w, h])
               	for ratio in aspect_ratios[i]:
                    w, h = (sizes[i]*sqrt(ratio)), (sizes[i]/sqrt(ratio))
                    anchor_boxes.append([(x+0.5)/feature_map_sizes[i], (y+0.5)/feature_map_sizes[i], w, h])
                    #print(anchor_boxes[-1])
                    w, h = (sizes[i]/sqrt(ratio)), (sizes[i]*sqrt(ratio))
                    anchor_boxes.append([(x+0.5)/feature_map_sizes[i], (y+0.5)/feature_map_sizes[i], w, h])
        self.default_boxes = torch.Tensor(anchor_boxes)
       # print(self.default_boxes)
    def iou(self, box1, box2):
        '''
        Compute the intersection over union of two set of boxes, each box is [x,y,w,h].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)
        lt = torch.max(
            (box1[:,:2].unsqueeze(1).expand(N,M,2)-  # [N,2] -> [N,1,2] -> [N,M,2]
            (box1[:,2:].unsqueeze(1).expand(N,M,2)/2)),  # [N,2] -> [N,1,2] -> [N,M,2]
            (box2[:,:2].unsqueeze(0).expand(N,M,2)-  # [M,2] -> [1,M,2] -> [N,M,2]
            (box2[:,2:].unsqueeze(0).expand(N,M,2)/2)) # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            (box1[:,:2].unsqueeze(1).expand(N,M,2)+  # [N,2] -> [N,1,2] -> [N,M,2]
            (box1[:,2:].unsqueeze(1).expand(N,M,2)/2)),  # [N,2] -> [N,1,2] -> [N,M,2]
            (box2[:,:2].unsqueeze(0).expand(N,M,2)+  # [N,2] -> [N,1,2] -> [N,M,2]
            (box2[:,2:].unsqueeze(0).expand(N,M,2)/2)) # [N,2] -> [N,1,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]) * (box1[:,3])  # [N,]
        area2 = (box2[:,2]) * (box2[:,3])  # [M,]

        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def iou2(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def encode(self, boxes, classes, threshold=0.5):
        '''Transform target bounding boxes and class labels to SSD boxes and classes.
        Match each object box to all the default boxes, pick the ones with the
        Jaccard-Index > 0.5:
            Jaccard(A,B) = AB / (A+B-AB)
        Args:
          boxes: (tensor) object bounding boxes (xmin,ymin,w,h) of a image, sized [#obj, 4].
          classes: (tensor) object class labels of a image, sized [#obj,].
          threshold: (float) Jaccard index threshold
        Returns:
          loc: (tensor) bounding boxes regression , sized [8732, 4].
          conf: (tensor) class labels, sized [8732,] if conf == 0, background else the label
        '''
        default_boxes = self.default_boxes
        num_default_boxes = default_boxes.size(0)
        num_objs = boxes.size(0)
        #print('boxs',boxes)
        #print('class',classes)
        iou = self.iou(  # [#obj,8732]
            boxes,
			default_boxes
        )

        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)        # [8732,]
        iou.squeeze_(0)            # [8732,]
        #print((iou>0.5).sum())
        boxes = boxes[max_idx]     # [8732,4]
        variances = [0.1, 0.2]
        cxcy = boxes[:,:2] - default_boxes[:,:2]  # [8732,2]
        cxcy /= variances[0] * default_boxes[:,2:]
        wh = boxes[:,2:] / default_boxes[:,2:]      # [8732,2]
        wh = torch.log(wh) / variances[1]
        loc = torch.cat([cxcy, wh], 1)  # [8732,4]

        conf = classes[max_idx]   # [8732,], background class = 0
        #print('jj',conf)
        #print('iou',iou)
        conf[iou<threshold] = 0       # background
        #print('con',(conf>0).sum())
        return loc, conf	
    def decode(self, loc, conf):
        '''
        loc [8732, 4]
        conf [9732, 2]
        return 
        boxes [ #, 4]
        labels [# , 1]
        '''
        #print('conf',conf)
        variances = (0.1, 0.2)
        wh = torch.exp(loc[:,2:]*variances[1]) * self.default_boxes[:,2:]
        cxcy = loc[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        box_preds = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)  # [8732,4]

        boxes = []
        labels = []
        scores = []
        num_classes = conf.size(1)
        #print(num_classes)
        for i in range(num_classes-1):
            score = conf[:,i+1]  # class i corresponds to (i+1) column
            mask = score > 0.1

            if not mask.any():
                continue

            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            if len(score) == 1:
                continue
            keep = self.nms(box, score, threshold=0.3)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        return boxes, labels, scores
    def nms(self, bboxes, scores, threshold=0.3, mode='union'):
        '''Non maximum suppression.
        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.
        Returns:
          keep: (tensor) selected indices.
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = (x2-x1) * (y2-y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 1: 
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return torch.LongTensor(keep)          
