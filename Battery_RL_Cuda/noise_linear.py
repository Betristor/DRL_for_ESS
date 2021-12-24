import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 
import math



class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)







# class NoisyLinear(nn.Linear):
# 	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
# 		super(NoisyLinear, self).__init__(in_features, out_features, bias=True)

# 		w = torch.full((out_features, in_features), sigma_init) 
# 		self.sigma_weight = nn.Parameter(w)
# 		z = torch.zeros(out_features, in_features)
# 		self.register_buffer("epsilon_weight", z)

# 		if bias:
# 			w = torch.full((out_features,), sigma_init)
# 			self.sigma_bias = nn.Parameter(w)
# 			z = torch.zeros(out_features)
# 			self.register_buffer("epsilon_bias", z)
# 		self.reset_parameters()

# 	def reset_parameters(self):
# 		std = math.sqrt(3 / self.in_features)
# 		self.weight.data.uniform_(-std, std)
# 		self.bias.data.uniform_(-std, std)

# 	def forward(self, input):
# 		self.epsilon_weight.normal_()
# 		bias = self.bias
# 		if bias is not None:
# 			self.epsilon_bias.normal_()
# 			bias = bias + self.sigma_bias * self.epsilon_bias.data
# 		v = self.sigma_weight * self.epsilon_weight.data + self.weight

# 		return F.linear(input, v, bias)



	# def reset_noise(self):
	# 	epsilon_in  = self._scale_noise(self.in_features)
	# 	epsilon_out = self._scale_noise(self.out_features)
	    
	# 	self.epsilon_weight.copy_(epsilon_out.ger(epsilon_in))
	# 	self.epsilon_bias.copy_(self._scale_noise(self.out_features))

	# def _scale_noise(self, size):
	# 	x = torch.randn(size)
	# 	x = x.sign().mul(x.abs().sqrt())
	# 	return x