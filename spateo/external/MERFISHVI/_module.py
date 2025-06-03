from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import (
    BaseMinifiedModeModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)
from scvi.utils import unsupported_if_adata_minified
from torch.nn.functional import one_hot

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class VAE(EmbeddingModuleMixin, BaseMinifiedModeModuleClass):
    """Variational auto-encoder :cite:p:`Lopez18`.

    Parameters
    ----------
    n_input
        Number of input features.
    n_batch
        Number of batches. If ``0``, no batch correction is performed.
    n_labels
        Number of labels.
    n_hidden
        Number of nodes per hidden layer. Passed into :class:`~scvi.nn.Encoder` and
        :class:`~scvi.nn.DecoderSCVI`.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers. Passed into :class:`~scvi.nn.Encoder` and
        :class:`~scvi.nn.DecoderSCVI`.
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        A list of integers containing the number of categories for each categorical covariate.
    dropout_rate
        Dropout rate. Passed into :class:`~scvi.nn.Encoder` but not :class:`~scvi.nn.DecoderSCVI`.
    dispersion
        Flexibility of the dispersion parameter when ``gene_likelihood`` is either ``"nb"`` or
        ``"zinb"``. One of the following:

        * ``"gene"``: parameter is constant per gene across cells.
        * ``"gene-batch"``: parameter is constant per gene per batch.
        * ``"gene-label"``: parameter is constant per gene per label.
        * ``"gene-cell"``: parameter is constant per gene per cell.
    log_variational
        If ``True``, use :func:`~torch.log1p` on input data before encoding for numerical stability
        (not normalization).
    gene_likelihood
        Distribution to use for reconstruction in the generative process. One of the following:

        * ``"nb"``: :class:`~scvi.distributions.NegativeBinomial`.
        * ``"zinb"``: :class:`~scvi.distributions.ZeroInflatedNegativeBinomial`.
        * ``"poisson"``: :class:`~scvi.distributions.Poisson`.
        * ``"normal"``: :class:`~torch.distributions.Normal`.
    latent_distribution
        Distribution to use for the latent space. One of the following:

        * ``"normal"``: isotropic normal.
        * ``"ln"``: logistic normal with normal params N(0, 1).
    encode_covariates
        If ``True``, covariates are concatenated to gene expression prior to passing through
        the encoder(s). Else, only gene expression is used.
    deeply_inject_covariates
        If ``True`` and ``n_layers > 1``, covariates are concatenated to the outputs of hidden
        layers in the encoder(s) (if ``encoder_covariates`` is ``True``) and the decoder prior to
        passing through the next layer.
    batch_representation
        ``EXPERIMENTAL`` Method for encoding batch information. One of the following:

        * ``"one-hot"``: represent batches with one-hot encodings.
        * ``"embedding"``: represent batches with continuously-valued embeddings using
          :class:`~scvi.nn.Embedding`.

        Note that batch representations are only passed into the encoder(s) if
        ``encode_covariates`` is ``True``.
    use_batch_norm
        Specifies where to use :class:`~torch.nn.BatchNorm1d` in the model. One of the following:

        * ``"none"``: don't use batch norm in either encoder(s) or decoder.
        * ``"encoder"``: use batch norm only in the encoder(s).
        * ``"decoder"``: use batch norm only in the decoder.
        * ``"both"``: use batch norm in both encoder(s) and decoder.

        Note: if ``use_layer_norm`` is also specified, both will be applied (first
        :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
    use_layer_norm
        Specifies where to use :class:`~torch.nn.LayerNorm` in the model. One of the following:

        * ``"none"``: don't use layer norm in either encoder(s) or decoder.
        * ``"encoder"``: use layer norm only in the encoder(s).
        * ``"decoder"``: use layer norm only in the decoder.
        * ``"both"``: use layer norm in both encoder(s) and decoder.

        Note: if ``use_batch_norm`` is also specified, both will be applied (first
        :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
    use_size_factor_key
        If ``True``, use the :attr:`~anndata.AnnData.obs` column as defined by the
        ``size_factor_key`` parameter in the model's ``setup_anndata`` method as the scaling
        factor in the mean of the conditional distribution. Takes priority over
        ``use_observed_lib_size``.
    use_observed_lib_size
        If ``True``, use the observed library size for RNA as the scaling factor in the mean of the
        conditional distribution.
    extra_payload_autotune
        If ``True``, will return extra matrices in the loss output to be used during autotune
    library_log_means
        :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of means of the log library sizes that
        parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
        ``use_observed_lib_size`` is ``False``.
    library_log_vars
        :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of variances of the log library sizes
        that parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
        ``use_observed_lib_size`` is ``False``.
    var_activation
        Callable used to ensure positivity of the variance of the variational distribution. Passed
        into :class:`~scvi.nn.Encoder`. Defaults to :func:`~torch.exp`.
    extra_encoder_kwargs
        Additional keyword arguments passed into :class:`~scvi.nn.Encoder`.
    extra_decoder_kwargs
        Additional keyword arguments passed into :class:`~scvi.nn.DecoderSCVI`.
    batch_embedding_kwargs
        Keyword arguments passed into :class:`~scvi.nn.Embedding` if ``batch_representation`` is
        set to ``"embedding"``.

    Notes
    -----
    Lifecycle: argument ``batch_representation`` is experimental in v1.2.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        extra_payload_autotune: bool = False,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        batch_embedding_kwargs: dict | None = None,
    ):
        from scvi.nn import DecoderSCVI, Encoder

        super().__init__()

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.extra_payload_autotune = extra_payload_autotune

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, " "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError("`dispersion` must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'.")

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding("batch", n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding("batch").embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            n_input_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """获取推断过程所需的输入张量。

        Args:
            tensors: 输入数据张量
            full_forward_pass: 是否执行完整的前向传播

        Returns:
            用于推断过程的输入字典
        """
        # 根据数据类型选择加载方式
        if full_forward_pass or self.minified_data_type is None:
            loader = "full_data"
        elif self.minified_data_type in [
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
            ADATA_MINIFY_TYPE.LATENT_POSTERIOR_WITH_COUNTS,
        ]:
            loader = "minified_data"
        else:
            raise NotImplementedError(f"未知的简化数据类型: {self.minified_data_type}")

        # 完整数据情况：提供表达数据和批次信息
        if loader == "full_data":
            return {
                "x": tensors["X"],  # 基因表达数据
                "batch_index": tensors["batch"],  # 批次索引
                "cont_covariates": tensors.get("continuous_covariates", None),  # 连续型协变量
                "cat_covariates": tensors.get("categorical_covariates", None),  # 分类型协变量
            }
        # 简化数据情况：提供已计算的潜在变量分布参数
        else:
            return {
                "qzm": tensors["scvi_latent_qzm"],  # 潜变量均值
                "qzv": tensors["scvi_latent_qzv"],  # 潜变量方差
                "observed_lib_size": tensors["observed_lib_size"],  # 观测到的文库大小
            }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """获取生成过程的输入张量。

        将推断步骤的输出与原始数据结合，准备生成网络的输入。

        Args:
            tensors: 原始数据张量
            inference_outputs: 推断过程的输出

        Returns:
            生成过程所需的输入字典
        """
        # 获取size_factor（如果提供）
        size_factor = tensors.get("size_factor", None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        return {
            "z": inference_outputs["z"],  # 潜在空间表示
            "library": inference_outputs["library"],  # 文库大小
            "batch_index": tensors["batch"],  # 批次索引
            "y": tensors["labels"],  # 细胞类型标签
            "cont_covariates": tensors.get("continuous_covariates", None),  # 连续型协变量
            "cat_covariates": tensors.get("categorical_covariates", None),  # 分类型协变量
            "size_factor": size_factor,  # 大小因子
        }

    def _compute_local_library_params(
        self,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算局部文库参数。

        为每个细胞计算文库大小的均值和方差参数，这些参数取决于细胞所属的批次。

        Args:
            batch_index: 形状为 (batch_size, 1) 的批次索引张量

        Returns:
            tuple: 包含两个张量，分别是文库大小对数的均值和方差
        """
        from torch.nn.functional import linear

        # 批次数量
        num_batches = self.library_log_means.shape[1]

        # 将批次索引转换为独热编码
        batch_one_hot = one_hot(batch_index.squeeze(-1), num_batches).float()

        # 计算每个细胞的文库大小对数均值
        # 相当于从全局文库均值表中查找对应批次的值
        library_log_means = linear(batch_one_hot, self.library_log_means)

        # 计算每个细胞的文库大小对数方差
        # 相当于从全局文库方差表中查找对应批次的值
        library_log_vars = linear(batch_one_hot, self.library_log_vars)

        return library_log_means, library_log_vars

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """运行常规推断过程，获取数据的潜在表示。

        Args:
            x: 基因表达数据
            batch_index: 批次索引
            cont_covs: 连续型协变量
            cat_covs: 分类型协变量
            n_samples: 采样数量

        Returns:
            包含潜在变量和分布的字典
        """
        x_ = x
        if self.use_observed_lib_size:
            # 计算观测到的文库大小（基因表达总和）
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            # 对数据取对数，增加数值稳定性
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates:
            # 将基因表达和连续协变量拼接起来
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            # 处理分类协变量
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if self.batch_representation == "embedding" and self.encode_covariates:
            # 使用嵌入表示批次
            batch_embedding = self.compute_embedding("batch", batch_index)
            encoder_input = torch.cat([encoder_input, batch_embedding], dim=-1)
            # 获取潜在变量分布和样本
            qz, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            # 使用独热编码表示批次
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                # 使用嵌入表示批次来编码文库大小
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                # 使用独热编码表示批次来编码文库大小
                ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded

        if n_samples > 1:
            # 多个样本情况下的处理
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))

        return {
            "z": z,  # 潜在空间表示
            "qz": qz,  # 潜在空间分布
            "ql": ql,  # 文库大小分布
            "library": library,  # 文库大小
        }

    @auto_move_data
    def _cached_inference(
        self,
        qzm: torch.Tensor,
        qzv: torch.Tensor,
        observed_lib_size: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | None]:
        """使用缓存的潜在变量分布参数进行推断。

        这种方法主要用于已经计算并存储了潜在变量分布的情况，
        可以加快推断速度，无需重新运行编码器网络。

        Args:
            qzm: 潜在变量均值
            qzv: 潜在变量方差
            observed_lib_size: 观测到的文库大小
            n_samples: 采样数量

        Returns:
            包含潜在变量和分布的字典
        """
        from torch.distributions import Normal

        # 创建潜在变量的正态分布
        latent_dist = Normal(qzm, qzv.sqrt())

        # 从分布中采样
        # 使用sample()而不是rsample()，因为我们不需要对z进行优化
        if n_samples == 1:
            untransformed_z = latent_dist.sample()
        else:
            untransformed_z = latent_dist.sample((n_samples,))

        # 变换潜在变量（如果使用的是logistic normal分布）
        z = self.z_encoder.z_transformation(untransformed_z)

        # 计算文库大小（取对数）
        library = torch.log(observed_lib_size)

        # 多样本情况下扩展文库大小
        if n_samples > 1:
            library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))

        return {
            "z": z,  # 潜在空间表示
            "qz": latent_dist,  # 潜在空间分布
            "ql": None,  # 没有文库大小分布（使用观测值）
            "library": library,  # 文库大小
        }

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process to get distribution parameters.

        This method takes latent representations and covariates to generate
        the parameters of the data distribution.

        Parameters
        ----------
        z : torch.Tensor
            Latent space representation
        library : torch.Tensor
            Library size factors
        batch_index : torch.Tensor
            Batch indices for each cell
        cont_covs : torch.Tensor, optional
            Continuous covariates
        cat_covs : torch.Tensor, optional
            Categorical covariates
        size_factor : torch.Tensor, optional
            Size factors (if not using library)
        y : torch.Tensor, optional
            Labels for each cell
        transform_batch : torch.Tensor, optional
            Batch to transform to (for batch correction)

        Returns
        -------
        dict
            Dictionary with distribution objects for data, library size, and latent space
        """
        from scvi.distributions import (
            NegativeBinomial,
            Normal,
            Poisson,
            ZeroInflatedNegativeBinomial,
        )
        from torch.nn.functional import linear

        # Prepare the decoder input by concatenating latent variables with covariates
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            # Handle case where dimensions don't match (e.g., when using multiple samples)
            decoder_input = torch.cat([z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        # Process categorical covariates if provided
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # For batch correction: transform to a specific batch if requested
        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        # Use library size as size factor if not explicitly provided
        if not self.use_size_factor_key:
            size_factor = library

        # Process batch information using either embedding or one-hot encoding
        if self.batch_representation == "embedding":
            # Get batch embedding and concatenate to decoder input
            batch_embedding = self.compute_embedding("batch", batch_index)
            decoder_input = torch.cat([decoder_input, batch_embedding], dim=-1)

            # Get parameters from decoder
            scale, dispersion_param, rate, dropout_prob = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                y,
            )
        else:
            # Standard approach using batch index directly
            scale, dispersion_param, rate, dropout_prob = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                batch_index,
                *categorical_input,
                y,
            )

        # Process dispersion parameter based on specified mode
        if self.dispersion == "gene-label":
            # Dispersion depends on cell label (cell type)
            dispersion_param = linear(one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r)
        elif self.dispersion == "gene-batch":
            # Dispersion depends on batch
            dispersion_param = linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            # One dispersion per gene
            dispersion_param = self.px_r

        # Ensure dispersion is positive by exponentiating
        dispersion_param = torch.exp(dispersion_param)

        # Create the appropriate distribution based on gene_likelihood
        if self.gene_likelihood == "zinb":
            # Zero-inflated negative binomial for sparse count data
            data_dist = ZeroInflatedNegativeBinomial(
                mu=rate,
                theta=dispersion_param,
                zi_logits=dropout_prob,
                scale=scale,
            )
        elif self.gene_likelihood == "nb":
            # Negative binomial for count data with overdispersion
            data_dist = NegativeBinomial(mu=rate, theta=dispersion_param, scale=scale)
        elif self.gene_likelihood == "poisson":
            # Poisson for count data
            data_dist = Poisson(rate=rate, scale=scale)
        elif self.gene_likelihood == "normal":
            # Normal distribution (experimental)
            data_dist = Normal(rate, dispersion_param, normal_mu=scale)

        # Set up prior distributions
        if self.use_observed_lib_size:
            # No library size prior if using observed library size
            lib_dist = None
        else:
            # Calculate parameters for library size prior based on batch
            lib_mean, lib_var = self._compute_local_library_params(batch_index)
            lib_dist = Normal(lib_mean, lib_var.sqrt())

        # Standard normal prior for latent space
        latent_dist = Normal(torch.zeros_like(z), torch.ones_like(z))

        # Return all distributions with clear names
        return {
            "gene_expression": data_dist,  # Distribution for gene expression
            "library_size": lib_dist,  # Distribution for library size
            "latent_space": latent_dist,  # Distribution for latent space
        }

    @unsupported_if_adata_minified
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: torch.tensor | float = 1.0,
    ) -> LossOutput:
        """计算变分自编码器的损失函数。

        损失函数由重构损失和KL散度两部分组成：
        1. 重构损失：衡量生成的数据与原始数据的匹配程度
        2. KL散度：衡量后验分布与先验分布的差异，起到正则化作用

        Args:
            tensors: 原始数据张量
            inference_outputs: 推断过程的输出
            generative_outputs: 生成过程的输出
            kl_weight: KL散度项的权重系数（用于KL退火）

        Returns:
            包含总损失和各部分损失的对象
        """
        from torch.distributions import kl_divergence

        # 获取原始基因表达数据
        x = tensors["X"]  # 使用直观的X代替REGISTRY_KEYS.X_KEY

        # 计算潜在变量的KL散度：后验分布q(z|x)与先验分布p(z)之间的差异
        kl_divergence_z = kl_divergence(inference_outputs["qz"], generative_outputs["latent_space"]).sum(dim=-1)

        # 计算文库大小的KL散度（如果使用学习的文库大小）
        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(inference_outputs["ql"], generative_outputs["library_size"]).sum(dim=1)
        else:
            # 如果使用观测的文库大小，则KL散度为0
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        # 计算重构损失：负对数似然
        reconstruction_loss = -generative_outputs["gene_expression"].log_prob(x).sum(-1)

        # 区分需要进行权重调整的KL散度（用于KL退火）
        kl_for_warmup = kl_divergence_z  # 潜在变量z的KL散度会参与退火
        kl_no_warmup = kl_divergence_l  # 文库大小l的KL散度不参与退火

        # 应用权重后的KL散度
        weighted_kl = kl_weight * kl_for_warmup + kl_no_warmup

        # 总损失 = 重构损失 + 加权KL散度
        total_loss = torch.mean(reconstruction_loss + weighted_kl)

        # 为自动调优准备额外的指标（如果需要）
        if self.extra_payload_autotune:
            extra_metrics = {
                "z": inference_outputs["z"],
                "batch": tensors["batch"],  # 使用直观的batch代替REGISTRY_KEYS.BATCH_KEY
                "labels": tensors["labels"],  # 使用直观的labels代替REGISTRY_KEYS.LABELS_KEY
            }
        else:
            extra_metrics = {}

        # 返回损失对象
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            kl_local={
                "kl_divergence_l": kl_divergence_l,  # 文库大小的KL散度
                "kl_divergence_z": kl_divergence_z,  # 潜在变量的KL散度
            },
            extra_metrics=extra_metrics,
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 1,
        max_poisson_rate: float = 1e8,
    ) -> torch.Tensor:
        r"""Generate predictive samples from the posterior predictive distribution.

        The posterior predictive distribution is denoted as :math:`p(\hat{x} \mid x)`, where
        :math:`x` is the input data and :math:`\hat{x}` is the sampled data.

        We sample from this distribution by first sampling ``n_samples`` times from the posterior
        distribution :math:`q(z \mid x)` for a given observation, and then sampling from the
        likelihood :math:`p(\hat{x} \mid z)` for each of these.

        Parameters
        ----------
        tensors
            Dictionary of tensors passed into :meth:`~scvi.module.VAE.forward`.
        n_samples
            Number of Monte Carlo samples to draw from the distribution for each observation.
        max_poisson_rate
            The maximum value to which to clip the ``rate`` parameter of
            :class:`~scvi.distributions.Poisson`. Avoids numerical sampling issues when the
            parameter is very large due to the variance of the distribution.

        Returns
        -------
        Tensor on CPU with shape ``(n_obs, n_vars)`` if ``n_samples == 1``, else
        ``(n_obs, n_vars,)``.
        """
        from scvi.distributions import Poisson

        inference_kwargs = {"n_samples": n_samples}
        _, generative_outputs = self.forward(tensors, inference_kwargs=inference_kwargs, compute_loss=False)

        dist = generative_outputs[MODULE_KEYS.PX_KEY]
        if self.gene_likelihood == "poisson":
            # TODO: NEED TORCH MPS FIX for 'aten::poisson'
            dist = (
                Poisson(torch.clamp(dist.rate.to("cpu"), max=max_poisson_rate))
                if self.device.type == "mps"
                else Poisson(torch.clamp(dist.rate, max=max_poisson_rate))
            )

        # (n_obs, n_vars) if n_samples == 1, else (n_samples, n_obs, n_vars)
        samples = dist.sample()
        # (n_samples, n_obs, n_vars) -> (n_obs, n_vars, n_samples)
        samples = torch.permute(samples, (1, 2, 0)) if n_samples > 1 else samples

        return samples.cpu()

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(
        self,
        tensors: dict[str, torch.Tensor],
        n_mc_samples: int,
        return_mean: bool = False,
        n_mc_samples_per_pass: int = 1,
    ):
        """Compute the marginal log-likelihood of the data under the model.

        Parameters
        ----------
        tensors
            Dictionary of tensors passed into :meth:`~scvi.module.VAE.forward`.
        n_mc_samples
            Number of Monte Carlo samples to use for the estimation of the marginal log-likelihood.
        return_mean
            Whether to return the mean of marginal likelihoods over cells.
        n_mc_samples_per_pass
            Number of Monte Carlo samples to use per pass. This is useful to avoid memory issues.
        """
        from torch import logsumexp
        from torch.distributions import Normal

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = []
        if n_mc_samples_per_pass > n_mc_samples:
            warnings.warn(
                "Number of chunks is larger than the total number of samples, setting it to the " "number of samples",
                RuntimeWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(
                tensors,
                inference_kwargs={"n_samples": n_mc_samples_per_pass},
                get_inference_input_kwargs={"full_forward_pass": True},
            )
            qz = inference_outputs[MODULE_KEYS.QZ_KEY]
            ql = inference_outputs[MODULE_KEYS.QL_KEY]
            z = inference_outputs[MODULE_KEYS.Z_KEY]
            library = inference_outputs[MODULE_KEYS.LIBRARY_KEY]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities
            p_z = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale)).log_prob(z).sum(dim=-1)
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x
            if n_mc_samples_per_pass == 1:
                log_prob_sum = log_prob_sum.unsqueeze(0)

            to_sum.append(log_prob_sum)
        to_sum = torch.cat(to_sum, dim=0)
        batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
        if return_mean:
            batch_log_lkl = torch.mean(batch_log_lkl).item()
        else:
            batch_log_lkl = batch_log_lkl.cpu()
        return batch_log_lkl

    @torch.inference_mode()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights in the linear decoder.

        Returns:
            numpy.ndarray: A matrix of weights showing how each latent dimension
                          contributes to each gene's expression level.
        """
        # Get the weights from the linear decoder
        # If batch normalization is used, we need to account for its effect
        if self.use_batch_norm is True:
            # Get the weight matrix from the first layer
            weights = self.decoder.factor_regressor.fc_layers[0][0].weight
            # Get the batch normalization layer
            batch_norm = self.decoder.factor_regressor.fc_layers[0][1]
            # Calculate the scaling factors from batch norm parameters
            variance = torch.sqrt(batch_norm.running_var + batch_norm.eps)
            gamma = batch_norm.weight
            scaling = gamma / variance
            # Create a diagonal matrix from the scaling factors
            scaling_matrix = torch.diag(scaling)
            # Apply the scaling to the weights
            loadings = torch.matmul(scaling_matrix, weights)
        else:
            # If no batch norm, just use the weights directly
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight

        # Convert to numpy array for easier downstream analysis
        loadings = loadings.detach().cpu().numpy()

        # If we have multiple batches, remove the batch effect columns
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        # The loadings matrix shows how each latent dimension influences each gene's expression
        return loadings


class LDVAE(VAE):
    """Linear-decoded Variational auto-encoder model.

    Implementation of :cite:p:`Svensson20`.

    This model uses a linear decoder, directly mapping the latent representation
    to gene expression levels. It still uses a deep neural network to encode
    the latent representation.

    Compared to standard VAE, this model is less powerful, but can be used to
    inspect which genes contribute to variation in the dataset. It may also be used
    for all scVI tasks, like differential expression, batch correction, imputation, etc.
    However, batch correction may be less powerful as it assumes a linear model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer (for encoder)
    n_latent
        Dimensionality of the latent space
    n_layers_encoder
        Number of hidden layers used for encoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_batch_norm
        Bool whether to use batch norm in decoder
    bias
        Bool whether to have bias term in linear decoder
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution.
    **kwargs
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        use_observed_lib_size: bool = False,
        **kwargs,
    ):
        from scvi.nn import Encoder, LinearDecoderSCVI

        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=use_observed_lib_size,
            **kwargs,
        )
        self.use_batch_norm = use_batch_norm
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
            return_dist=True,
        )
        self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            use_batch_norm=use_batch_norm,
            use_layer_norm=False,
            bias=bias,
        )

    @torch.inference_mode()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights in the linear decoder.

        Returns:
            numpy.ndarray: A matrix of weights showing how each latent dimension
                          contributes to each gene's expression level.
        """
        # Get the weights from the linear decoder
        # If batch normalization is used, we need to account for its effect
        if self.use_batch_norm is True:
            # Get the weight matrix from the first layer
            weights = self.decoder.factor_regressor.fc_layers[0][0].weight
            # Get the batch normalization layer
            batch_norm = self.decoder.factor_regressor.fc_layers[0][1]
            # Calculate the scaling factors from batch norm parameters
            variance = torch.sqrt(batch_norm.running_var + batch_norm.eps)
            gamma = batch_norm.weight
            scaling = gamma / variance
            # Create a diagonal matrix from the scaling factors
            scaling_matrix = torch.diag(scaling)
            # Apply the scaling to the weights
            loadings = torch.matmul(scaling_matrix, weights)
        else:
            # If no batch norm, just use the weights directly
            loadings = self.decoder.factor_regressor.fc_layers[0][0].weight

        # Convert to numpy array for easier downstream analysis
        loadings = loadings.detach().cpu().numpy()

        # If we have multiple batches, remove the batch effect columns
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        # The loadings matrix shows how each latent dimension influences each gene's expression
        return loadings
