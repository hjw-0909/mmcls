_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/lanes.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': 1})
# model = dict(
#     head=dict(
#     num_classes = 4,
#     loss = dict(loss_weight = [4.47, 19.64, 25.92, 1])
# )
#      )
model = dict(
    head=dict(
    num_classes = 4,
    loss = dict(loss_weight = [1, 4.4, 5.8, 0.2])
)
     )