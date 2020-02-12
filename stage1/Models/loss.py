import torch
import torch.nn as nn

class WeightedMSE:
	def __init__(self, b_mult = 2.0):
		self.b_mult = b_mult

	def forward(self, y_true, y_pred):
		epsilon = 1e-07

		b = torch.mean(y_true, dim=(1, 2, 3)) * self.b_mult
		y_pred[y_pred<epsilon] = epsilon
		y_pred[y_pred>(1-epsilon)] = 1-epsilon

		zeros = torch.zeros(y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]).cuda()
		ones = torch.ones(y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]).cuda()
		
		# background covering
		zero_idx = torch.where(y_true==0, ones, zeros)
		zero_idx.type_as(torch.cuda.FloatTensor())
		m_neg = torch.mean(((y_pred ** 2) * zero_idx), dim=(1, 2, 3))

		# edges covering
		non_zero_idx = torch.where(y_true!=0, ones, zeros)
		non_zero_idx.type_as(torch.cuda.FloatTensor())
		m_pos = torch.mean(((y_pred - y_true) ** 2) * non_zero_idx, dim=(1, 2, 3))

		m = (1-b) * m_pos + b * m_neg
		return torch.mean(m, dim=-1)

if __name__=="__main__":
	loss = WeightedMSE()
	y_pred = torch.FloatTensor([[[[1, 0, 2, 2],[1, 1, 1, 1]],[[2, 2, 2, 2],[1, 4, 2, 2]]], [[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]]]).cuda()
	print(y_pred.shape)
	print(loss.forward(y_pred, y_pred))
		
		

		
