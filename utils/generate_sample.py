from model.gan import Generator
import torch
from  torchvision import utils
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    model_path = "/home/xiao/mywork/mnist-classify/output/CGAN/0.1/2023-07-03-1688381356.0/model.pt"
    device = 'cpu'
    G = Generator(100, 64, 10, 1)
    G.load_state_dict(torch.load(model_path, map_location=device))
    # G = torch.load(model_path, map_location=device)

    fake_each = 1000

    label2hot = torch.eye(10).reshape(10, 10, 1, 1).to(device)
    fixed_noise = torch.randn(fake_each*10, 100, 1, 1).to(device)
    fixed_label = label2hot[torch.arange(10).repeat(fake_each).sort().values]

    with torch.no_grad():
        fake = G(fixed_noise, fixed_label).detach().cpu()
    print(f"result shape: {fake.shape}") # (10000, 1, 32, 32)

    # save
    save_root = "/home/xiao/mywork/mnist-classify/data/MNIST/png/"
    csv_root = "/home/xiao/mywork/mnist-classify/csv/"
    csv_list = []
    counter = 0
    for idx in range(0, len(fake)):
        label = idx // fake_each
        print(f"Coping idx {idx}, label {label}")
        img = fake[idx]
        save_file = f"fake_{label}/mnist_{counter}-{label}.png"
        counter += 1
        if not os.path.exists(save_root+f"fake_{label}"):
            os.makedirs(save_root+f"fake_{label}")
        utils.save_image(img, save_root+save_file, normalize=True)
        csv_list.append(np.array([save_file, str(label)]))
        # if counter == 10:
        #     break
    fake_samples = np.array(csv_list) # (n, 2)
    np.savetxt(fname=csv_root + "train_0.1_generation.csv", X=fake_samples, fmt="%s", delimiter=',')
    real_samples = pd.read_csv(csv_root+"train_0.1_classify.csv", header=None).values # (m, 2)
    new_samples = np.concatenate([real_samples, fake_samples], axis=0)
    print(f"new samples: {new_samples.shape}")
    np.savetxt(fname=csv_root + "train_0.1_add.csv", X=new_samples, fmt="%s", delimiter=',')