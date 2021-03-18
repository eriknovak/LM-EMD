import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel


def get_cost_matrix(q_embeds: torch.Tensor, d_embeds: torch.Tensor):
    """Calculates the cost matrix of the embeddings

    Args:
        q_embeds: The query embeddings tensor.
        d_embeds: The documents embeddings tensor.
    Returns:
        torch.Tensor: The cost matrix between the query and document embeddings.

    """
    # normalize the embeddings
    q_embeds = f.normalize(q_embeds, p=2, dim=2)
    d_embeds = f.normalize(d_embeds, p=2, dim=2)
    # calculate and return the cost matrix
    cost_matrix = q_embeds.matmul(d_embeds.transpose(1, 2))
    cost_matrix = torch.ones_like(cost_matrix) - cost_matrix
    return cost_matrix


def get_distributions(attention: torch.FloatTensor):
    """Generates the distribution tensor

    Args:
        attention: The attention tensor.

    Returns:
        torch.Tensor: The distribution tensor.

    """
    dist = torch.ones_like(attention) * attention
    dist = dist / dist.sum(dim=1).view(-1, 1).repeat(1, attention.shape[1])
    dist.requires_grad = False
    return dist


def sinkhorn(
    q_dist: torch.FloatTensor,
    d_dist: torch.FloatTensor,
    cost_matrix: torch.Tensor,
    reg: float,
    nit: int,
):
    """Documentation
    The sinkhorn algorithm adapted for PyTorch from the
        PythonOT library <https://pythonot.github.io/>.

    Args:
        q_dist: The queries distribution.
        d_dist: The documents distribution.
        cost_matrix: The cost matrix.
        reg: The regularization factor.
        nit: Number of maximum iterations.

    Returns:
        torch.Tensor: The transportation matrix.

    """
    # asset the dimensions
    assert (
        q_dist.shape[0] == cost_matrix.shape[0]
        and q_dist.shape[1] == cost_matrix.shape[1]
    )
    assert (
        d_dist.shape[0] == cost_matrix.shape[0]
        and d_dist.shape[1] == cost_matrix.shape[2]
    )
    # prepare the initial variables
    K = torch.exp(-cost_matrix / reg)
    Kp = (1 / q_dist).reshape(q_dist.shape[0], -1, 1) * K
    # initialize the u and v tensor
    u = torch.ones_like(q_dist)
    v = torch.ones_like(d_dist)
    istep = 0
    while istep < nit:
        # calculate K.T * u for each example in batch
        KTransposeU = K.transpose(1, 2).bmm(u.unsqueeze(2)).squeeze()
        # calculate the v_{i} tensor
        v = d_dist / KTransposeU
        # calculate the u_{i} tensor
        u = 1.0 / Kp.bmm(v.unsqueeze(2)).squeeze()
        # go to next step
        istep = istep + 1
    # calculate the transport matrix
    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    return U.bmm(K).bmm(V)


def prep_relevant_embeds(embeds: torch.Tensor, attention_mask: torch.Tensor):
    """Prepares the token embeddings for mean and max types

    Args:
        embeds: The token embeddings.
        attention_mask: The attention mask.

    Returns:
        torch.Tensor: The tensor with padding embeddings set to zero.

    """
    return embeds * attention_mask.unsqueeze(2).repeat(1, 1, embeds.shape[2])


class BERT(nn.Module):
    def __init__(self, type: str = "emd", reg: float = 1, nit: int = 500):
        """The BERT model adapted for document ranking.

        Args:
            type: The rank calculation type (Default: "emd").
            reg: The regularization factor used with "emd" (Default: 1).
            nit: The maximum number of iterations used with "emd" (Default: 500).
        """
        super(BERT, self).__init__()
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.type = type
        self.reg = reg
        self.nit = nit

    def forward(
        self,
        q_input_ids: torch.LongTensor = None,
        q_attention_mask: torch.LongTensor = None,
        d_input_ids: torch.LongTensor = None,
        d_attention_mask: torch.LongTensor = None,
    ):
        """Calculate the similarities between the queries and documents.

        Args:
            q_input_ids: The query input IDs tensor.
            q_attention_mask: The query attention mask tensor.
            d_input_ids: The document input IDs tensor.
            d_attention_mask: The document attention mask tensor.
        """
        # get query embeddings
        q_embeds = self.model(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
        )["last_hidden_state"]

        # get document embeddings
        d_embeds = self.model(
            input_ids=d_input_ids,
            attention_mask=d_attention_mask,
        )["last_hidden_state"]

        if self.type == "emd":
            # get the cost matrix
            C = get_cost_matrix(q_embeds, d_embeds)
            # get query and texts distributions
            q_dist = get_distributions(q_attention_mask.float())
            d_dist = get_distributions(d_attention_mask.float())
            # solve the optimal transport problem
            T = sinkhorn(q_dist, d_dist, C, self.reg, self.nit)
            # calculate the distances
            distances = (C * T).view(C.shape[0], -1).sum(dim=1)
            # return the loss, transport and cost matrices
            return distances, C, T
        elif self.type == "cls":
            # get the first [CLS] token
            q_embeds = q_embeds[:, 0, :]
            d_embeds = d_embeds[:, 0, :]
        elif self.type == "max":
            # get the maximum values of the embeddings
            q_embeds, _ = prep_relevant_embeds(q_embeds, q_attention_mask).max(dim=1)
            d_embeds, _ = prep_relevant_embeds(d_embeds, d_attention_mask).max(dim=1)
        elif self.type == "mean":
            # get the mean of the embeddings
            q_embeds = prep_relevant_embeds(q_embeds, q_attention_mask).mean(dim=1)
            d_embeds = prep_relevant_embeds(d_embeds, d_attention_mask).mean(dim=1)

        # normalize the vectors before calculating
        q_embeds = f.normalize(q_embeds, p=2, dim=1)
        d_embeds = f.normalize(d_embeds, p=2, dim=1)

        # calculate the mean distances
        distances = q_embeds.matmul(d_embeds.T)
        distances = distances.new_ones(distances.shape) - distances
        # calculate the loss value
        return distances, None, None
