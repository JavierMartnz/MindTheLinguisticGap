import torch.nn as nn
from torch.nn import init


class _DomainSpecificBatchNorm(nn.ModuleList):
    def __init__(self, num_features, num_domains=1, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self._domain_index = 0
        self._num_domains = num_domains
        self._set_batch_norm(num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                             track_running_stats=True)

    def reset_running_stats(self):
        for m in self:
            if m.track_running_stats:
                m.running_mean.zero_()
                m.running_var.fill_(1)
                m.num_batches_tracked.zero_()

    def reset_parameters(self):
        for m in self:
            m.reset_running_stats()
            if m.affine:
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def __repr__(self):
        return self._get_name() + '({num_features}, num_domains={num_domains}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                                  'track_running_stats={track_running_stats})[{domain_index}]'.format(
            num_domains=self._num_domains, domain_index=self.domain_index, **self[0].__dict__)

    @property
    def num_domains(self):
        return self._num_domains

    @property
    def domain_index(self):
        return self._domain_index

    @domain_index.setter
    def domain_index(self, value):
        if value in range(self._num_domains):
            self._domain_index = value
        else:
            raise IndexError(f"Invalid domain index {value}")

    def forward(self, input):
        return self[self._domain_index](input)

    def _set_batch_norm(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                        track_running_stats=True):
        raise NotImplementedError

    @classmethod
    def set_domain_index(cls, modules, domain_index):
        if not isinstance(modules, (tuple, list)):
            modules = [modules, ]
        for module in modules:
            if isinstance(module, _DomainSpecificBatchNorm):
                module.domain_index = domain_index
            for name, child in module.named_children():
                cls.set_domain_index(child, domain_index)


class DomainSpecificBatchNorm1d(_DomainSpecificBatchNorm):
    def _set_batch_norm(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                        track_running_stats=True):
        self.extend([nn.BatchNorm1d(num_features, eps, momentum, affine,
                                    track_running_stats) for _ in range(self.num_domains)])


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _set_batch_norm(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                        track_running_stats=True):
        self.extend(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(self.num_domains)])


class DomainSpecificBatchNorm3d(_DomainSpecificBatchNorm):
    def _set_batch_norm(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                        track_running_stats=True):
        self.extend(
            [nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats) for _ in range(self.num_domains)])


class SyncDomainSpecificBatchNorm(_DomainSpecificBatchNorm):
    def _set_batch_norm(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                        track_running_stats=True):
        self.extend(
            [nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in
             range(self.num_domains)])


set_domain_index = _DomainSpecificBatchNorm.set_domain_index
