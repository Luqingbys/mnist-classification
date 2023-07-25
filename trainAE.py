import argparse
import torch
from torch import nn
import torch.optim as optim
from torchvision.utils import save_image
from model.ae import *
import os
import shutil
import numpy as np


def save_checkpoint(state, is_best, outdir):
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
	best_file = os.path.join(outdir, 'model_best.pth')
	torch.save(state, checkpoint_file)
	if is_best:
		shutil.copyfile(checkpoint_file, best_file)


def loss_func(recon_x, inputs):
	reconstruction_loss = F.binary_cross_entropy(recon_x, inputs, reduction='sum')
	#reconstruction_loss = F.l1_loss(recon_x, inputs, reduction='mean')
	# divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1.0 - log_sigma)
	# loss = reconstruction_loss + divergence
	return reconstruction_loss


def train_ae(net: nn.Module, z_dim, batch_size, trainloader, validloader, epochs, valid_every, output, device='cpu'):
	optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
	start_epoch = 0
	if not os.path.exists(output+'/reconstructed'):
		os.makedirs(output+'/reconstructed')
	if not os.path.exists(output+'/random_sample'):
		os.makedirs(output+'/random_sample')
	if not os.path.exists(output+'/weight'):
		os.makedirs(output+'/weight')

	# training
	for epoch in range(start_epoch, epochs):
		for i, data in enumerate(trainloader):
            
			inputs = data[0].to(device)
			recon = net(inputs)

			recon_loss = loss_func(recon, inputs)
			# zero out the paramter gradients
			optimizer.zero_grad()
			recon_loss.backward()
			optimizer.step()

			# print statistics every 100 batches
			if (i + 1) % 10 == 0:
				print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
					  .format(epoch + 1, epochs, i + 1, len(trainloader), recon_loss.item()))

			if i == 0:
				# 重构图像可视化
				x_concat = torch.cat([inputs.view(-1, 1, 28, 28), res.view(-1, 1, 28, 28)], dim=3)
				save_image(x_concat, ("./%s/reconstructed/epoch-%d.png" % (output, epoch + 1)))

		# valid
		if (epoch + 1) % valid_every == 0:
			valid_avg_loss = 0.0
			best_valid_loss = np.inf
			with torch.no_grad():
				for idx, valid_data in enumerate(validloader):
					valid_inp = valid_data[0].to(device)
					# forward
					valid_res = net(valid_inp)
					valid_recon_loss = loss_func(valid_res, valid_inp)
					valid_avg_loss += valid_recon_loss

				valid_avg_loss /= len(validloader.dataset)
				print('valid loss:', valid_avg_loss.item())

				# randomly sample some images' latent vectors from its distribution
				z = torch.randn(batch_size, z_dim).to(device)
				random_res = net.decode(z).view(-1, 1, 28, 28)
				save_image(random_res, "./%s/random_sample/epoch-%d.png" % (output, epoch + 1))

				# save model
				is_best = valid_avg_loss < best_valid_loss
				best_valid_loss = min(valid_avg_loss, best_valid_loss)
				if is_best:
					with open(output+'/weight/model_best.pth', 'wb') as f:
						torch.save(net, f)
	
	with open(output+'/weight/model_final.pth', 'wb') as f:
		torch.save(net, f)
	print('Train finished!')
