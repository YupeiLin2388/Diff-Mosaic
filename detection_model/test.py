# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc

def ro_curve(y_pred, y_label, figure_file):
    y_label = np.array(y_label)
    y_pred = np.array(y_pred) 
    print(y_label,y_pred)   
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred) 
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(figure_file)
    
class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load(args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        pred_list = []
        y_label =[]
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    #pred_list.extend(pred.detach().cpu().numpy().reshape(-1))
                    #y_label.extend(labels.detach().cpu().numpy().reshape(-1))
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                    #pred_listl.append(pred.detach().cpu().numpy())
                    #y_label.append(labels.detach().cpu().numpy())
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU = self.mIoU.get()
            Recalls=[a for a in recall]
            Precisions=[a for a in precision]
            FA, PD = self.PD_FA.get(len(val_img_ids))
            #print('tr,fr')
            print(mean_IOU)
            tr=[a for a in ture_positive_rate]
            fa=[b for b in false_positive_rate]
            print(args.model_dir.split('/'))
            np.save(args.model_dir.split('/')[2]+'.npy',np.array([PD,FA,tr,fa]))
            print('FA=',[a for a in FA])
            print('PD=',[b for b in PD])
            print(tr,fa)


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





