import time
import torch
import torch.nn as nn
from torchvision import utils
import os
from utils.draw import drawLoss
from torchvision.utils import save_image


def train_gan(G: nn.Module, D: nn.Module, optimizerG, optimizzerD, trainloader, epochs, device, output, img_size=32):
    if not os.path.exists(output + '/fake'):
        os.makedirs(output + '/fake')
    G_losses = []
    D_losses = []
    D_x_list = []
    D_z_list = []
    img_list = []
    loss_tep = 10

    REAL_LABEL = 1.
    FAKE_LABEL = 0.
    label2hot = torch.eye(10).reshape(10, 10, 1, 1).to(device)

    label_fills = torch.zeros(10, 10, img_size, img_size)
    for i in range(10):
        label_fills[i][i] = torch.ones(img_size, img_size)
    label_fills = label_fills.to(device)

    fixed_noise = torch.randn(100, 100, 1, 1).to(device)
    fixed_label = label2hot[torch.arange(10).repeat(10).sort().values]

    criterion = nn.BCELoss()
    print("Start training!")
    for epoch in range(epochs):
        start_time = time.time()
        for i, data in enumerate(trainloader):
            # TODO training Discriminator with true dataset, namely binary classify task
            D.zero_grad()
            real_img = data[0].to(device) # true images
            b = real_img.shape[0] # batch_size
            real_labels = torch.full((b,), REAL_LABEL).to(device)    # true images label, 1
            fake_labels = torch.full((b,), FAKE_LABEL).to(device)   # fake images label, 0

            D_label = label_fills[data[1]]    # 训练判别器的数据标签,
            G_label = label2hot[data[1]]    # 训练生成器的数据标签, e.g. [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

            out = D(real_img, D_label).view(-1)
            # real_labels = real_labels.to(torch.float)
            error_D_real = criterion(out, real_labels)
            error_D_real.backward()
            D_x = out.mean().item() # 对真实图像的判定结果


            # TODO training Discriminator with fake dataset, namely binary classify task
            noise = torch.randn(b, 100, 1, 1).to(device)
            fake = G(noise, G_label) # 生成假样本
            out = D(fake.detach(), D_label).view(-1)
            # fake_labels = fake_labels.to(torch.float)
            error_D_fake = criterion(out, fake_labels)
            error_D_fake.backward()
            D_G_z1 = out.mean().item() # 对假图像的判定结果
            loss_D = error_D_real + error_D_fake
            optimizzerD.step()


            # TODO update Generator parameters
            G.zero_grad()
            out = D(fake, D_label).view(-1)
            loss_G = criterion(out, real_labels)     # 生成器的损失函数是判别器对假样本的判别结果和真实样本标签（1）之间的差异，也就是说，生成器更希望判别器将假样本（fake）都判定为真实样本（real_labels）
            loss_G.backward()
            D_G_z2 = out.mean().item()
            optimizerG.step()

            end_time = time.time()
            run_time = round(end_time - start_time)
            print(f"Epoch: [{epoch+1}/{epochs}], Step: [{i+1}/{len(trainloader)}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, D(x): {D_x:.4f}, D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}], Time: {run_time}", end='\r')
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            D_x_list.append(D_x)
            D_z_list.append(D_G_z2)

            if loss_G < loss_tep:
                torch.save(G.state_dict(), output+'/model.pt')
                loss_tep = loss_G

        # Check how the generator is doing by saving G's output on fixed_noise and fixed_label
        with torch.no_grad():
            fake = G(fixed_noise, fixed_label).detach().cpu()
        fake_img = utils.make_grid(fake, nrow=10)
        save_image(fake_img, ("./%s/fake/epoch-%d.png" % (output, epoch + 1)))
        img_list.append(utils.make_grid(fake, nrow=10))

    drawLoss(G_losses, output, file="train_Generator_loss")
    drawLoss(D_losses, output, file="train_discriminator_loss")
    drawLoss(D_x_list, output, file="D(x)")
    drawLoss(D_z_list, output, file="D(G(z))")