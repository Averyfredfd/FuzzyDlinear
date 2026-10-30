import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models import Autoformer, DLinear, FuzzyDLinear, Informer, Transformer
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.noise_injector import NoiseInjector  # <--- 新增：导入噪声注入器

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # ========== 修复：局部定义 model_dict ==========
        model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'FuzzyDLinear': FuzzyDLinear,
            'Informer': Informer,
            'Transformer': Transformer,
        }

        # 通过局部变量 model_dict 获取对应模型，不再调用 self.model_dict
        model = model_dict[self.args.model].Model(self.args).float()
        # =================================================

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # 计算模型参数量并打印输出
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'=' * 50}")
        print(f"Model Structure: {self.args.model}")
        print(f"Trainable Parameters: {total_params / 1e6:.4f} M")
        print(f"{'=' * 50}\n")

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _build_noise_injector(self):
        """训练/验证/测试共用同一套噪声参数，避免评估口径不一致。"""
        return NoiseInjector(
            noise_type=getattr(self.args, 'noise_type', 'none'),
            intensity=getattr(self.args, 'noise_intensity', 0.0),
            spike_ratio=getattr(self.args, 'spike_ratio', 0.03),
            spike_amp=getattr(self.args, 'spike_amp', None)
        )

    def vali(self, vali_data, vali_loader, criterion, noise_injector=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # 验证集注入与训练同分布的噪声，否则噪声训练时早停/选模标准会错位
                if noise_injector is not None and noise_injector.noise_type != 'none':
                    batch_x = noise_injector(batch_x)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder (验证时不注入噪声，评估干净数据表现)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ====== 初始化噪声注入器（train / vali / test 同分布）======
        noise_injector = self._build_noise_injector()
        if noise_injector.noise_type != 'none':
            print(f"\n[Noise Injector] Activated: type={noise_injector.noise_type}, "
                  f"intensity={noise_injector.intensity}, spike_ratio={noise_injector.spike_ratio}, "
                  f"spike_amp={noise_injector.spike_amp}\n")
        # ====================================

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # ====== 核心：仅对输入注入噪声，标签保持干净 ======
                if noise_injector.noise_type != 'none':
                    batch_x = noise_injector(batch_x)
                # ==================================================

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, noise_injector)
            test_loss = self.vali(test_data, test_loader, criterion, noise_injector)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            noise_injector = self._build_noise_injector()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ====== 恶劣环境模拟：对输入特征注入噪声 ======
                if noise_injector.noise_type != 'none':
                    batch_x = noise_injector(batch_x)
                # ============================================

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 测试时不注入噪声，评估模型在干净数据上的真实预测能力
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # ========== 新增/修改：反归一化并计算核心指标 (MSE, MAE) ==========
        # 当 target 经过归一化训练时，必须 inverse_transform 才能得到有物理意义的误差
        if test_data.scale and getattr(self.args, 'inverse', False):
            shape = preds.shape
            preds = test_data.inverse_transform(preds.squeeze(0)).reshape(shape)
            trues = test_data.inverse_transform(trues.squeeze(0)).reshape(shape)

        # 核心指标计算
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)

        print(f"\n{'=' * 50}")
        print(f"Test Execution -> Model: {self.args.model} | Seq: {self.args.seq_len} | Pred: {self.args.pred_len}")
        print(f"Final Metrics -> MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        print(f"{'=' * 50}\n")
        # ====================================================================

        # 结果保存
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return