from b import Ui_MainWindow
from PyQt5 import QtWidgets
from dataset import SyntheticCellDataset
from model import UNet
from loss import DiceLoss
import torch
from utils import save_checkpoint

class Abc_Def(Ui_MainWindow):

    def __init__(self, MainWindow):
        super(Abc_Def, self).__init__(MainWindow)
        self.mask_dir = ''
        self.mask_dir = ''
        self.image.clicked.connect(self.pick_image_folder)
        self.mask.clicked.connect(self.pick_mask_folder)
        self.train_start.clicked.connect(self.start_train)
        self.epoch_num = self.epoch.value()
        self.lr_num = self.lr.value()

    def clicked(self, var):
        print(self.epoch_num)
        print(var)

    def pick_image_folder(self):
        dialog = QtWidgets.QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Folder")
        self.image_dir = folder_path
        self.train_logs.setText(folder_path)

    def pick_mask_folder(self):
        dialog = QtWidgets.QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Folder")
        self.mask_dir = folder_path
        self.train_logs.setText(folder_path)

    def train(self, model, train_loader, device, optimizer):
        model.train()
        steps = len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        train_loss = 0.0
        dsc_loss = DiceLoss()

        for i, data in enumerate(train_loader):
            x,y = data

            optimizer.zero_grad()
            y_pred = model(x.to(device))
            loss = dsc_loss(y_pred, y.to(device))
            print(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
        return model, train_loss/len(train_loader), optimizer

    def validate(self, model, val_loader, device):

        with torch.no_grad():
            model.eval()
            val_loss = 0.0

            for i, data in enumerate(val_loader):
                x,y = data

                y_pred = model(x.to(device))
                loss = dsc_loss(y_pred, y.to(device))

                val_loss += loss.item()
        return val_loss/len(val_loader)

    def start_train(self):
        if(self.image_dir!='' or self.mask_dir!=''):
            dataset = SyntheticCellDataset(self.image_dir, self.mask_dir)
            indices = torch.randperm(len(dataset)).tolist()
            sr = int(0.2 * len(dataset))
            train_set = torch.utils.data.Subset(dataset, indices[:-sr])
            val_set = torch.utils.data.Subset(dataset, indices[-sr:])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=False, pin_memory=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device)
            model = UNet()
            model.to(device)
            print('ethi atkuchi')


            optimizer = torch.optim.Adam(model.parameters(), self.lr_num)

            val_overall = 1000
            for epoch in range(self.epoch_num):
                print('ethi atkuchi')
                model, train_loss, optimizer = self.train(model, train_loader, device, optimizer)
                val_loss = self.validate(model, val_loader, device)

                # if val_loss < val_overall:
                #     save_checkpoint(args.model_save_dir + '/epoch_'+str(epoch+1), model, train_loss, val_loss, epoch)
                #     val_overall = val_loss

                print('[{}/{}] train loss :{} val loss : {}'.format(epoch+1, num_epoch, train_loss, val_loss))
            print('Training completed')
        else:
            print('Select mask and image directories')




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Abc_Def(MainWindow)
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
