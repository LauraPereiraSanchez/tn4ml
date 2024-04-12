from typing import Any, Collection, Optional, Sequence, Tuple, Callable
from tqdm import tqdm
import funcy
from time import time
import numpy as np

import quimb.tensor as qtn
import quimb as qu
import autoray
import optax
import jax
from flax.training.early_stopping import EarlyStopping

from ..embeddings import *
from ..strategy import *
from ..util import gradient_clip

from jax import config
config.update("jax_enable_x64", True)

class Model(qtn.TensorNetwork):
    """:class:`tn4ml.models.Model` class models training model of class :class:`quimb.tensor.tensor_core.TensorNetwork`.

    Attributes
    ----------
    loss : `Callable`, or `None`
        Loss function. See :mod:`tn4ml.loss` for examples.
    strategy : :class:`tn4ml.strategy.Strategy`
        Strategy for computing gradients.
    optimizer : str
        Type of optimizer matching names of optimizers from optax.
    """

    def __init__(self):
        """Constructor method for :class:`tn4ml.models.Model` class."""
        self.loss: Callable = None
        self.strategy : Any = 'global'
        self.optimizer : optax.GradientTransformation = optax.adam
        self.learning_rate : float = 1e-2
        self.train_type : str = 'unsupervised'
        self.gradient_transforms : Sequence = None
        self.opt_state : Any = None
        self.cache : dict = {}

    def save(self, model_name, dir_name='~'):

        """Saves :class:`tn4ml.models.Model` to pickle file.

        Parameters
        ----------
        model_name : str
            Name of Model.
        dir_name: str
            Directory for saving Model.
        """
        exec(compile('from ' + self.__class__.__module__ + ' import ' + self.__class__.__name__, '<string>', 'single'))
        arrays = tuple(map(lambda x: np.asarray(jax.device_get(x)), self.arrays))
        model = type(self)(arrays)
        qu.save_to_disk(model, f'{dir_name}/{model_name}.pkl')

    def configure(self, **kwargs):

        """Configures :class:`tn4ml.models.Model` for training setting the arguments.

        Parameters
        ----------
        kwargs : dict
            Arguments for configuration.
        """
        for key, value in kwargs.items():
            if key == "strategy":
                if isinstance(value, Strategy):
                    self.strategy = value
                elif value in ['sweeps', 'local', 'dmrg', 'dmrg-like']:
                    self.strategy = Sweeps()
                elif value in ['global']:
                    self.strategy = 'global'
                else:
                    raise ValueError(f'Strategy "{value}" not found')
            elif key in ["optimizer", "loss", "train_type", "learning_rate", "gradient_transforms"]:
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found")
                
        if self.train_type not in ['unsupervised', 'supervised']:
            raise AttributeError("Specify type of training: 'supervised' or 'unsupervised'!")
        
        if not hasattr(self, 'optimizer') or not hasattr(self, 'gradient_transforms'):
            raise AttributeError("Provide 'optimizer' or sequence of 'gradient_transforms'! ")
        
        if self.gradient_transforms:
            self.optimizer = optax.chain(*self.gradient_transforms)
        else:
            if hasattr(self, 'optimizer'):
                self.optimizer = self.optimizer(learning_rate = self.learning_rate)
            else:
                self.optimizer = optax.adam(learning_rate=self.learning_rate)

    def update_tensors(self, params, sitetags=None):

        """Updates tensors of the model with new parameters.
        
        Parameters
        ----------
        params : sequence of :class:`jax.numpy.ndarray`
            New parameters of the model.
        sitetags : sequence of str, or default `None`
            Names of tensors for differentiation (for Sweeping strategy).
        
        Returns
        -------
        None
        """
        if isinstance(self.strategy, Sweeps):
            if sitetags is None:
                raise ValueError("For Sweeping strategy you must provide names of tensors for differentiation.")
            tensor = self.select_tensors(sitetags)[0]
            tensor.modify(data = params[0])
        else:
            for tensor, array in zip(self.tensors, params):
                tensor.modify(data=array)

    def create_cache(self,
                    model,
                    embedding: Embedding,
                    input_shape: tuple,
                    target_shape: Optional[tuple] = None,
                    inputs_dtype: Any = jnp.float_,
                    targets_dtype: Any = None):
        """Creates cache for compiled functions to calculate loss and gradients.
        
        Parameters
        ----------
        model : :class:`tn4ml.models.Model`
            Model to train.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        input_shape : tuple
            Shape of input data.
        target_shape : tuple, or default `None`
            Shape of target data.
        inputs_dtype : Any
            Data type of input data.
        targets_dtype : Any, or default `None`
            Data type of target data.

        Returns
        -------
        None
        """
        
        if self.strategy == 'global':
            params = model.arrays

            dummy_input = jnp.ones(shape=input_shape, dtype=inputs_dtype)
            if self.train_type == 'unsupervised':
                def loss_fn(data, *params):
                    with qtn.contract_backend("jax"), autoray.backend_like("jax"):
                        tn = model.copy()
                        for tensor, array in zip(tn.tensors, params):
                            tensor.modify(data=array)
                        input_tn = embed(data, embedding)
                        return self.loss(tn, input_tn)
                loss_ir = jax.jit(jax.vmap(loss_fn, in_axes=[0] + [None]*self.L)).lower(dummy_input, *params)
                grads_ir = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=(i + 1 for i in range(self.L))), in_axes=[0] + [None]*self.L)).lower(dummy_input, *params)
            else:
                dummy_targets = jnp.ones(shape=target_shape, dtype=targets_dtype)
                def loss_fn(data, targets, *params):
                    with qtn.contract_backend("jax"), autoray.backend_like("jax"):
                        tn = model.copy()
                        for tensor, array in zip(tn.tensors, params):
                            tensor.modify(data=array)
                        input_tn = embed(data, embedding)
                        return self.loss(tn, input_tn, targets)
                loss_ir = jax.jit(jax.vmap(loss_fn, in_axes=[0, 0] + [None]*self.L)).lower(dummy_input, dummy_targets, *params)
                grads_ir = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=(i + 2 for i in range(self.L))), in_axes=[0, 0] + [None] * self.L)).lower(dummy_input, dummy_targets, *params)

            # save compiled functions
            self.cache["loss_compiled"] = loss_ir.compile()
            self.cache["grads_compiled"] = grads_ir.compile()
            self.cache["hash"] = hash((embedding, self.strategy, self.loss, self.train_type, self.optimizer, self.shape))
        else:
            raise ValueError('Only supports creating cache for global gradient descent strategy!')
        
    def create_train_step(self, params, loss_func, grads_func):

        """Creates function for calculating value and gradients of loss, and function for one step in training procedure.
        Initializes the optimizer and creates optimizer state.

        Parameters
        ----------
        params : sequence of :class:`jax.numpy.ndarray`
            Parameters of the model.
        cache_loss : sequence or dict
            Cache of compiled functions to calculate loss value.
        cache_grad : sequence or dict
            Cache of compiled functions to calculate gradient of loss function.
        
        Returns
        -------
        train_step : function
            Function to perform one training step.
        opt_state : tuple
            State of optimizer at the initialization. 
        """

        init_params = {
            i: jnp.array(data)
            for i, data in enumerate(params)
        }
        opt_state = self.optimizer.init(init_params)

        def value_and_grad(params, data, targets = None):
            """ Calculates loss value and gradient.

            Parameters
            ----------
            params : sequence of :class:`jax.numpy.ndarray`
                Parameters of the model.
            data : sequence of :class:`jax.numpy.ndarray`
                Input data.
            targets : sequence of :class:`jax.numpy.ndarray` or None
                Target data (if training is supervised).

            Returns
            -------
            float, :class:`jax.numpy.ndarray`
            """
            if targets is not None:
                l = loss_func(data, targets, *params)
                g = grads_func(data, targets, *params)
            else:
                l = loss_func(data, *params)
                g = grads_func(data, *params)
            
            g = [jnp.sum(gi, axis=0) / data.shape[0] for gi in g]
            return jnp.sum(l)/data.shape[0], g

        def train_step(params, opt_state, data, sitetags=None, grad_clip_threshold=None):
            """ Performs one training step.

            Parameters
            ----------
            params : sequence of :class:`jax.numpy.ndarray`
                Parameters of the model.
            opt_state : tuple
                State of optimizer.
            data : sequence of :class:`jax.numpy.ndarray`
                Input data.
            sitetags : sequence of str
                Names of tensors for differentiation (for Sweeping strategy).
            
            Returns
            -------
            float, :class:`jax.numpy.ndarray`
            """

            if len(data) == 2:
                data, targets = data
                data, targets = jnp.array(data), jnp.array(targets)
            else:
                data = jnp.array(data)
                targets = None

            loss, grads = value_and_grad(params, data, targets)
            
            if grad_clip_threshold:
                grads = gradient_clip(grads, grad_clip_threshold)
            
            # convert to pytree structure
            grads = {i: jnp.array(data)
                    for i, data in enumerate(grads)}
            params = {i: jnp.array(data)
                    for i, data in enumerate(params)}

            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # convert back to arrays
            params = tuple(jnp.array(v) for v in params.values())

            # update TN inplace
            self.update_tensors(params, sitetags)
            
            self.normalize()

            return params, opt_state, loss
        
        return train_step, opt_state

    def train(self,
            inputs: Collection,
            val_inputs: Optional[Collection] = None,
            targets: Optional[Collection] = None,
            val_targets: Optional[Collection] = None,
            batch_size: Optional[int] = None,
            epochs: Optional[int] = 1,
            embedding: Embedding = trigonometric(),
            normalize: Optional[bool] = False,
            canonize: Optional[Tuple] = tuple([False, None]),
            time_limit: Optional[int] = None,
            earlystop: Optional[EarlyStopping] = None,
            # callbacks: Optional[Sequence[Tuple[str, Callable]]] = None,
            gradient_clip_threshold: Optional[float] = None,
            cache: Optional[bool] = True,
            dtype: Any = jnp.float_):
        
        """Performs the training procedure of :class:`tn4ml.models.Model`.

        Parameters
        ----------
        inputs : sequence of :class:`numpy.ndarray`
            Data used for training procedure.
        val_inputs : sequence of :class:`numpy.ndarray`
            Data used for validation.
        targets: sequence of :class:`numpy.ndarray`
            Targets for training procedure (if training is supervised).
        val_targets: sequence of :class:`numpy.ndarray`
            Targets for validation (if training is supervised).
        batch_size : int, or default `None`
            Number of samples per gradient update.
        epochs : int
            Number of epochs for training.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        normalize : bool
            If True, the model is normalized after each iteration.
        canonize: tuple([bool, int])
            tuple indicating is model canonized after each iteration. Example: (True, 0) - model is canonized in canonization center = 0.
        time_limit: int
            Time limit on model's training in seconds.
        earlystop : :class:` flax.training.early_stopping.EarlyStopping`
            Early stopping training when monitored metric stopped improving.
        gradient_clip_threshold : float
            Threshold for gradient clipping.
        cache : bool
            If True, cache compiled functions for loss and gradients.
        
        Returns
        -------
        history: dict
            Records training loss and metric values.
        """
        
        if cache and not self.strategy == 'global':
            raise ValueError("Caching is only supported for global gradient descent strategy!")

        if cache and canonize[0]:
            raise ValueError("Caching is not supported for canonization, because canonization can change shapes of tensors!")
        
        if targets is not None:
            if targets.ndim == 1:
                targets = np.expand_dims(targets, axis=-1)

        self.batch_size = batch_size

        n_batches = (len(inputs)//self.batch_size)

        self.history = dict()
        self.history['loss'] = []
        self.history['epoch_time'] = []
        self.history['unfinished'] = False
        if val_inputs is not None:
            self.history['val_loss'] = []

        if cache:
            # Caching loss computation and gradients
            if not 'hash' in self.cache or self.cache["hash"] != hash((embedding, self.strategy, self.loss, self.train_type, self.optimizer, self.shape)):
                model = self.copy()
                self.create_cache(model,
                                embedding,
                                (self.batch_size, inputs.shape[1]),
                                (self.batch_size, targets.shape[1]) if targets is not None else None,
                                dtype,
                                targets.dtype if targets is not None else None)
                
                # initialize optimizer - only important to get opt_state
                params = self.arrays
                self.step, self.opt_state = self.create_train_step(params=params, loss_func=self.cache['loss_compiled'], grads_func=self.cache['grads_compiled'])
        else:
            # Train without caching
            model = self.copy()

            if isinstance(self.strategy, Sweeps):
                if self.train_type == 'unsupervised':
                    def loss_fn(data, params):
                        with qtn.contract_backend("jax"):
                            tn = model.copy()
                            tn.select_tensors(sitetags)[0].modify(data=params)
                            input_tn = embed(data, embedding)
                            return self.loss(tn, input_tn)
                    self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, None]))
                    self.grads_func = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=[1]), in_axes=[0, None]))
                else:
                    def loss_fn(data, targets, params):
                        with qtn.contract_backend("jax"):
                            tn = model.copy()
                            tn.select_tensors(sitetags)[0].modify(data=params)
                            input_tn = embed(data, embedding)
                            return self.loss(tn, input_tn, targets)
                    self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, 0, None]))
                    self.grads_func = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=[2]), in_axes=[0, 0, None]))
                
                # initialize optimizer
                self.opt_states = []

                for s, sites in enumerate(self.strategy.iterate_sites(self)):
                    self.strategy.prehook(self, sites)
                    
                    sitetags = [self.site_tag(site) for site in sites]
                    
                    params_i = self.select_tensors(sitetags)[0].data
                    params_i = jnp.expand_dims(params_i, axis=0)
                    self.step, opt_state = self.create_train_step(params=params_i, loss_func=self.loss_func, grads_func=self.grads_func)

                    self.opt_states.append(opt_state)

                    self.strategy.posthook(self, sites)
            else:
                if self.strategy != 'global':
                    raise ValueError("Only Global Gradient Descent and DMRG Sweeping strategy is supported for now!")
                
                if self.train_type == 'unsupervised':
                    def loss_fn(data, *params):
                        with qtn.contract_backend("jax"):
                            tn = model.copy()
                            for tensor, array in zip(tn.tensors, params):
                                tensor.modify(data=array)
                            input_tn = embed(data, embedding)
                            return self.loss(tn, input_tn)
                    self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0] + [None]*self.L))
                    self.grads_func = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=(i + 1 for i in range(self.L))), in_axes=[0] + [None]*self.L))
                else:
                    def loss_fn(data, targets, *params):
                        with qtn.contract_backend("jax"):
                            tn = model.copy()
                            for tensor, array in zip(tn.tensors, params):
                                tensor.modify(data=array)
                            input_tn = embed(data, embedding)
                            return self.loss(tn, input_tn, targets)
                    self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, 0] + [None]*self.L))
                    self.grads_func = jax.jit(jax.vmap(jax.grad(loss_fn, argnums=(i + 2 for i in range(self.L))), in_axes=[0, 0] + [None] * self.L))
                
                # initialize optimizer
                params = self.arrays
                self.step, self.opt_state = self.create_train_step(params=params, loss_func=self.loss_func, grads_func=self.grads_func)
        
        start_train = time()
        with tqdm(total = epochs, desc = "epoch") as outerbar:
            for epoch in range(epochs):
                time_epoch = time()
                loss_batch = 0
                for batch_data in _batch_iterator(inputs, targets, self.batch_size, dtype=dtype):
                    if isinstance(self.strategy, Sweeps):
                        loss_curr = 0
                        for s, sites in enumerate(self.strategy.iterate_sites(self)):
                            self.strategy.prehook(self, sites)
                            
                            sitetags = [self.site_tag(site) for site in sites]
                            
                            params_i = self.select_tensors(sitetags)[0].data
                            params_i = jnp.expand_dims(params_i, axis=0) # add batch dimension
                            
                            _, self.opt_states[s], loss_group = self.step(params_i, self.opt_states[s], batch_data, sitetags=sitetags, grad_clip_threshold=gradient_clip_threshold)
                            
                            self.strategy.posthook(self, sites)

                            loss_curr += loss_group
                        loss_curr /= (s+1)
                    else:
                        params = self.arrays
                        _, self.opt_state, loss_curr = self.step(params, self.opt_state, batch_data, grad_clip_threshold=gradient_clip_threshold)

                    loss_batch += loss_curr

                    if normalize:
                        self.normalize()

                    if canonize[0]:
                        self.canonize(canonize[1])

                loss_epoch = loss_batch/n_batches

                self.history['loss'].append(loss_epoch)

                self.history['epoch_time'].append(time() - time_epoch)

                # if for some reason you have a limited amount of time to train the model
                if time_limit is not None and (time() - start_train + np.mean(self.history['epoch_time']) >= time_limit):
                    self.history["unfinished"] = True
                    return self.history
                
                # evaluate validation loss
                if val_inputs is not None:
                    loss_val_epoch = self.evaluate(val_inputs, val_targets, dtype=dtype)
                    self.history['val_loss'].append(loss_val_epoch)

                    # early stopping
                    if earlystop:
                        earlystop = earlystop.update(loss_val_epoch)
                        if earlystop.should_stop:
                            print(f'Met early stopping criteria, breaking at epoch {epoch}')
                            break
                else:
                    if earlystop:
                        earlystop = earlystop.update(loss_epoch)
                        if earlystop.should_stop:
                            print(f'Met early stopping criteria on training data, breaking at epoch {epoch}')
                            break
                
                outerbar.update()
                outerbar.set_postfix({'loss': loss_epoch, 'val_loss': self.history['val_loss'][-1] if 'val_loss' in self.history else None})
                

        return self.history
    
    def predict(self, sample: Collection, embedding: Embedding = trigonometric(), return_tn: bool = False):
        """Predicts the output of the model.
        
        Parameters
        ----------
        sample : :class:`numpy.ndarray`
            Input data.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        return_tn : bool   
            If True, returns tensor network, otherwise returns data. Useful when you want to vmap over predict function.
        
        Returns
        -------
        :class:`quimb.tensor.tensor_core.TensorNetwork`
            Output of the model.
        """
        assert sample.ndim == 1, "Input data must be 1D array!"
        
        if len(np.squeeze(sample)) < self.L:
            raise ValueError(f"Input data must have at least {self.L} elements!")
        
        tn_sample = embed(sample, embedding)

        if callable(getattr(self, "apply", None)):
            output = self.apply(tn_sample)
        else:
            output = self & tn_sample

            for ind in tn_sample.outer_inds():
                output.contract_ind(ind=ind)
            if not return_tn:
                output = output^all
        
        if return_tn:
            return output
        else:
            return output.arrays

    def evaluate(self, 
                 inputs: Collection,
                 targets: Optional[Collection] = None,
                 batch_size: Optional[int] = 1,
                 embedding: Embedding = trigonometric(),
                 evaluate_type: str = 'supervised',
                 return_list: bool = False,
                 loss_function: Optional[Callable] = None,
                 dtype: Any = jnp.float_):
        
        """Evaluates the model on the data.

        Parameters
        ----------
        inputs : sequence of :class:`numpy.ndarray`
            Data used for evaluation.
        targets: sequence of :class:`numpy.ndarray`
            Targets for evaluation (if evaluation is supervised).
        batch_size : int, or default `None`
            Number of samples per evaluation.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        evaluate_type : str
            Type of evaluation: 'supervised' or 'unsupervised'.
        return_list : bool
            If True, returns list of loss values for each batch.
        dtype : Any
            Data type of input data.
        
        Returns
        -------
        float
            Loss value.
        """

        if targets is not None:
            if targets.ndim == 1:
                targets = np.expand_dims(targets, axis=-1)
        
        if not hasattr(self, 'batch_size') and len(self.cache.keys()) == 0:
            self.batch_size = batch_size

        if len(inputs) < self.batch_size and self.cache["loss_compiled"] is not None:
            raise ValueError(f"Input data must have at least {self.batch_size} samples, because loss is precompiled!")
        
        if not hasattr(self, 'loss') or self.loss is None:
            self.loss = loss_function
        
        loss_value = 0
        if return_list:
            loss = []
        
        if (len(self.cache.keys()) == 0 or batch_size is None) and hasattr(self, 'batch_size'):
            batch_size = self.batch_size

        for batch_data in _batch_iterator(inputs, targets, batch_size, dtype=dtype):
            if len(batch_data) == 2:
                x, y = batch_data
                x, y = jnp.array(x), jnp.array(y)
            else:
                x = jnp.array(batch_data)
                y = None

            if isinstance(self.strategy, Sweeps):
                if not hasattr(self, 'loss_func'):
                    if evaluate_type == 'unsupervised':
                        def loss_fn(data, params):
                            with qtn.contract_backend("jax"):
                                tn = self.copy()
                                tn.select_tensors(sitetags)[0].modify(data=params)
                                input_tn = embed(data, embedding)
                                return self.loss(tn, input_tn)
                        self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, None]))
                    else:
                        def loss_fn(data, targets, params):
                            with qtn.contract_backend("jax"):
                                tn = self.copy()
                                tn.select_tensors(sitetags)[0].modify(data=params)
                                input_tn = embed(data, embedding)
                                return self.loss(tn, input_tn, targets)
                        self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, 0, None]))
                
                loss_curr = np.zeros((x.shape[0],))
                for s, sites in enumerate(self.strategy.iterate_sites(self)):
                    self.strategy.prehook(self, sites)
                    
                    sitetags = [self.site_tag(site) for site in sites]
                    
                    params_i = self.select_tensors(sitetags)[0].data
                    params_i = jnp.expand_dims(params_i, axis=0)

                    if evaluate_type == 'unsupervised':
                        loss_group = self.loss_func(x, *params_i)
                    else:
                        loss_group = self.loss_func(x, y, *params_i)

                    self.strategy.posthook(self, sites)

                    loss_curr += loss_group
                loss_curr /= (s+1)
            else:
                params = self.arrays
                if len(self.cache.keys()) == 0 or (len(self.cache.keys()) > 0 and batch_size != self.batch_size):
                    if not hasattr(self, 'loss_func'):
                        if evaluate_type == 'unsupervised':
                            def loss_fn(data, *params):
                                with qtn.contract_backend("jax"):
                                    tn = self.copy()
                                    for tensor, array in zip(tn.tensors, params):
                                        tensor.modify(data=array)
                                    input_tn = embed(data, embedding)
                                    return self.loss(tn, input_tn)
                            self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0] + [None]*self.L))
                        else:
                            def loss_fn(data, targets, *params):
                                with qtn.contract_backend("jax"):
                                    tn = self.copy()
                                    for tensor, array in zip(tn.tensors, params):
                                        tensor.modify(data=array)
                                    input_tn = embed(data, embedding)
                                    return self.loss(tn, input_tn, targets)
                            self.loss_func = jax.jit(jax.vmap(loss_fn, in_axes=[0, 0] + [None]*self.L))
                    if evaluate_type == 'unsupervised':
                        loss_curr = self.loss_func(x, *params)
                    else:
                        loss_curr = self.loss_func(x, y, *params)
                else:
                    if evaluate_type == 'unsupervised':
                        loss_curr = self.cache["loss_compiled"](x, *params)
                    else:
                        loss_curr = self.cache["loss_compiled"](x, y, *params)
            
            loss_value += np.mean(loss_curr)
            if return_list:
                loss.extend(loss_curr)

        if return_list:
            return np.asarray(loss)
        
        return loss_value / (len(inputs)//self.batch_size)
    
    def convert_to_pytree(self):
        """Converts tensor network to pytree structure and returns its skeleon.
        Reference to :function:`quimb.tensor.pack`.
        
        Returns
        -------
        pytree (dict)
        skeleton (Tensor, TensorNetwork, or similar) – A copy of obj with all references to the original data removed.
        """
        params, skeleton = qtn.pack(self)
        return params, skeleton

def load_model(model_name, dir_name='~'):
    """Loads the Model from pickle file.

    Parameters
    ----------
    model_name : str
        Name of the model.
    dir_name : str
        Directory where model is stored.
    
    Returns
    -------
    :class:`tn4ml.models.Model` or subclass
    """

    return qu.load_from_disk(f'{dir_name}/{model_name}.pkl')

def _check_chunks(chunked: Collection, batch_size: int = 2):
    """Checks if the last chunk has lower size then batch size.
    
    Parameters
    ----------
    chunked : sequence
        Sequence of chunks.
    batch_size : int
        Size of batch.
    
    Returns
    -------
    sequence
    """
    if len(chunked[-1]) < batch_size:
        chunked = chunked[:-1]
    return chunked

def _batch_iterator(x: Collection, y: Optional[Collection] = None, batch_size:int = 2, dtype: Any = jnp.float_):
    """Iterates over batches of data.
    
    Parameters
    ----------
    x : sequence
        Input data.
    batch_size : int
        Size of batch.
    y : sequence, or default `None`
        Target data.
    dtype : Any
        Data type of input data.
    
    Yields
    ------
    tuple
        Batch of input and target data (if target data is provided)
    """
        
    x_chunks = funcy.chunks(batch_size, jax.numpy.asarray(x, dtype=dtype))
    x_chunks = _check_chunks(list(x_chunks), batch_size)

    if y is not None:
        y_chunks = funcy.chunks(batch_size, jax.numpy.asarray(y)) # dont change dtype
        y_chunks = _check_chunks(list(y_chunks), batch_size)

        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            yield x_chunk, y_chunk
    else:
        for x_chunk in x_chunks:
            yield x_chunk    