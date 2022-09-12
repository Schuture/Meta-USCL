import torch
import torch.nn.functional as F
import numpy as np


class InfoNCE(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(InfoNCE, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _get_similarity_function(self, use_cosine_similarity):
        ''' Cosine similarity or dot similarity for computing loss '''
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size) # I(2Nx2N), identity matrix
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size) # lower diagonal matrix, N non-zero elements
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size) # upper diagonal matrix, N non-zero elements
        mask = torch.from_numpy((diag + l1 + l2)) # [2N, 2N], with 4N elements are non-zero
        mask = (1 - mask).type(torch.bool) # [2N, 2N], with 4(N^2 - N) elements are 'True'
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # extend the dimensions and calculate similarity
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C), N input samples
        # y shape: (1, 2N, C), 2N output representations
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0)) # extend the dimensions before calculating similarity 
        return v

    def forward(self, zl, zr, weights):
        if len(zl.shape) == 1:
            zl, zr = zl.unsqueeze(0), zr.unsqueeze(0)
            
        if self.batch_size != zl.shape[0]:
            self.batch_size = zl.shape[0] # the last batch may not have the same batch size
        
        if weights != None and len(weights.shape) != 1:
            weights = weights.squeeze()
        
        try:
            zl = F.normalize(zl, dim=1) # [N, C], normalize each feature vector
            zr = F.normalize(zr, dim=1)
        except:
            print(zl.shape)
            raise ValueError
    
        # mask for negative samples
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        representations = torch.cat([zr, zl], dim=0) # [N, C] => [2N, C]

        # similarity matrix for 2N samples in a batch
        similarity_matrix = self.similarity_function(representations, representations) # [2N, 2N]

        l_pos = torch.diag(similarity_matrix, self.batch_size) # upper diagonal, N x [left, right] positive sample pairs
        r_pos = torch.diag(similarity_matrix, -self.batch_size) # lower diagonal, N x [right, left] positive sample pairs
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1) # similarity of positive pairs, [2N, 1]

        # similarity of positive pairs, [2N, 2N-1]
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits = logits / self.temperature

        # labels are all 0, meaning the first value of each vector is the nominator term of CELoss
        # each denominator contains 2N+1-2 = 2N-1 terms, corresponding to all similarities between the sample and other samples.
        labels = torch.zeros(2 * self.batch_size).to(self.device).long() 
        loss = self.criterion(logits, labels)
        
        if weights == None:
            weights = torch.ones(2 * self.batch_size) / (2 * self.batch_size)
            weights = weights.to('cuda')
        else:
            # normalize to avoid learning 0 weights, and make the weights sum to 1
            weights = torch.cat((weights, weights))
            _sum = torch.sum(weights)
            weights = weights / _sum
            weights = weights.to('cuda')

        loss = torch.dot(loss, weights) # weighted loss

        return loss
















