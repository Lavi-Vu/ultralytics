import torch 
from torch import nn 

# class FC(nn.Layer):
#     def __init__(self, embedding_size, class_num):
#         super(FC, self).__init__()
#         self.class_num = class_num
#         self.embedding_size = embedding_size
#         weight_attr = torch.nn.initializer.XavierUniform()
#         bias_attr = None
#         self.fc = nn.Linear(
#             self.embedding_size,
#             self.class_num,
#             weight_attr=weight_attr,
#             bias_attr=bias_attr)
        
#     def forward(self, input, label=None):
#         x = self.fc(input)
#         return x
    
# class ArcMargin(nn.Layer):
#     def __init__(self,
#                  embedding_size,
#                  class_num,
#                  margin=0.5,
#                  scale=80.0,
#                  easy_margin=False):
#         super().__init__()
#         self.embedding_size = embedding_size
#         self.class_num = class_num
#         self.margin = margin
#         self.scale = scale
#         self.easy_margin = easy_margin
#         self.weight = self.create_parameter(
#             shape=[self.embedding_size, self.class_num],
#             is_bias=False,
#             default_initializer=torch.nn.initializer.XavierNormal())