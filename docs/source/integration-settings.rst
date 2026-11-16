Integration settings
====================

In the last two tutorials, we have discussed how the MadNIS integrator can be used for single- and
multi-channel integration. This page describes further options to adapt the integrator for
different applications.

Changing the integration domain
-------------------------------

The default integration domain in MadNIS is the unit hypercube :math:`[0,1]^d`. To choose different
finite intervals for the integration domain, a custom
:py:class:`Integrand <madnis.integrator.Integrand>` object can be created. For instance, to
integrate the function from the first tutorial over the interval :math:`[0,2]^4`, we can
construct the integrator as follows:

.. code-block:: python

    from madnis.integrator import Integrator, Integrand
    f = Integrand(lambda x: (2 * x).prod(dim=1), input_dim=4, bounds=[[0.0, 2.0]] * 4)
    integrator = Integrator(f)

For an infinite integration domain we can change the normalizing flow in our integrator to use a
normal latent space instead of the default uniform latent space. For example, to integrate the
function :math:`f(x,y) = \exp(- x^2 - y^2)`, we can define our integrator as

.. code-block:: python

    integrator = Integrator(
        lambda x: x.square().sum(dim=1).neg().exp(),
        dims=2,
        flow_kwargs={"uniform_latent": False},
    )

Note that this only works well if the means and standard deviations of our integrand are roughly
zero and one, respectively. Otherwise, the integrand should be shifted and rescaled. Also note that,
internally, the integration interval is limited to :math:`[-10, 10]` if the flow is built this way.
An alternative way to integrate over the whole space of real numbers is to apply a logit
transformation and the corresponding Jacobian in the integrand function.

Network architecture
--------------------

The default values for the normalizing flow hyperparameters in MadNIS will work well for many
simple, low-dimensional integrands. For more complex functions, it can be necessary to build a
larger network. This can be done with the ``flow_kwargs`` argument of the
:py:class:`Integrator <madnis.integrator.Integrator>` class. They are passed on to the constructor
of the :py:class:`Flow <madnis.nn.Flow>` class. By default, the flow is built with a sufficient
number of coupling blocks such that every component is conditioned on every other component at least
once. Hence, it is normally not necessary to set the number of coupling blocks by hand. In the
following example, we change the settings for the depth and number of hidden nodes of the flow
sub-networks.

.. code-block:: python

    integrator = Integrator(
        lambda x: (2 * x).prod(dim=1),
        dims = 4,
        flow_kwargs={"layers": 4, "units": 64},
    )

Similarly, the ``cwnet_kwargs`` parameter can be used to change the hyperparameters of the network
used for the trained channel weights.

For even more flexibility with the network architecture, the ``flow`` and ``cwnet`` arguments can
be used to replace the normalizing flows and channel weight networks used by default in MadNIS.
The interface that objects passed to the ``flow`` arguments have to support is specified by the
abstract base class :py:class:`Distribution <madnis.integrator.Distribution>`. A class used as
channel weight network should have a ``forward`` function that accepts tensors of shape
``(batch_size, remapped_dim)`` and returns tensors of shape ``(batch_size, channel_count)``. The
output of the network is then added to the logarithm of the prior channel weights if they were
provided. After that, the normalized channel weights are computed.

Training hyperparameters
------------------------

There are several hyperparameters that affect the network training that can be set when the
constructor of :py:class:`Integrator <madnis.integrator.Integrator>` is called. The loss function
can be changed using the ``loss`` argument. By default, the KL divergence will be used for
single-channel integration. The integral variance and reverse KL divergence are available as
alternative options. The same options are available for multi-channel training, however only the
variance loss allows for the simultaneous optimization of channel mappings and weights.

Further important training parameters are the batch size (``batch_size`` argument) and the learning
rate (``learning_rate`` argument). To enable training with a variable learning rate, a learning rate
scheduler has to be constructed. This can be done by defining a function that returns the scheduler
with the optimizer as a parameter. For instance, the following code sets cosine annealing as the
learning rate scheduling.

.. skip: next

.. code-block:: python

    from torch.optim.lr_scheduler import CosineAnnealingLR
    integrator = Integrator(
        ..., # other arguments
        scheduler = lambda opt: CosineAnnealingLR(opt, n_steps) # number of training iterations
    )

If a learning rate scheduler is given, the learning rate used for the current training iteration
will be given in the :py:class:`TrainingStatus <madnis.integrator.TrainingStatus>` object.
Similarly, we can also set the optimizer by passing a function that constructs the optimizer given
the trainable parameters. For instance, to use the ``SGD`` optimizer instead of ``Adam``, we can use

.. skip: next

.. code-block:: python

    from torch.optim import SGD
    integrator = Integrator(
        ..., # other arguments
        optimizer = lambda params: SGD(params, lr=1e-3)
    )

Batches with outlier gradients can destabilize the training. To limit their effect, the gradients
can be clipped before every optimization step with the ``grad_clip_threshold`` argument, as done in
MadGraph7. The gradients of all trained parameters are rescaled such that their global norm does not
exceed ``grad_clip_threshold * sqrt(n)``, where ``n`` is the number of trained parameters. In other
words, the root mean square of the gradient components is at most ``grad_clip_threshold``.
MadGraph7 uses

.. skip: next

.. code-block:: python

    integrator = Integrator(
        ..., # other arguments
        grad_clip_threshold = 0.003
    )

By default, the gradients are not clipped.

VEGAS pre-training
------------------

VEGAS is a commonly used algorithm for importance sampling that works by assuming a factorized
distribution, i.e. no correlations between different dimensions. It then models the one-dimensional distributions using variable-width bins with uniform probabilities. This makes training VEGAS much
faster than the training of a neural network using stochastic gradient descent. We can use that to
our advantage in MadNIS by training a VEGAS grid first and then using it to initialize our
normalizing flow, using the :py:class:`VegasPreTraining <madnis.integrator.VegasPreTraining>` class.
It is constructed using an :py:class:`Integrator <madnis.integrator.Integrator>` instance and uses
the same integrand, sample buffer and integration cache. It relies on the ``vegas`` package which is
an optional dependency of MadNIS. The pre-training can be performed with the following code:

.. code-block:: python

    from madnis.integrator import VegasPreTraining

    # Construct integrator here

    vegas = VegasPreTraining(integrator, bins=64, damping=0.8)
    vegas.train([1000,2000,4000])
    vegas.initialize_integrator()

    # Regular MadNIS training here

The only two hyperparameters are the number of bins for the VEGAS grid and the damping parameter
that influences the VEGAS convergence (high: fast adaption, low: stable convergence). The parameters
of the train method specify the number of samples in each VEGAS iteration (per channel in the
multi-channel case). The last line initializes the normalizing flow in the integrator.

Similar to the :py:class:`Integrator <madnis.integrator.Integrator>` class, the
:py:class:`VegasPreTraining <madnis.integrator.VegasPreTraining>` class also has methods ``sample``,
``integrate``, ``integration_metrics`` and ``unweighting_metrics`` which directly draw samples using
the VEGAS grid. This allows us to compare the VEGAS performance and the normalizing flow performance
directly.

Dealing with zeros
------------------

The MadNIS integrator has two parameters that control how it treats samples with an integrand value
of zero. The parameter ``batch_size_threshold`` controls the minimum fraction of samples with
non-zero integrand in every channel during the training. If the fraction of non-zero samples in a
channel, estimated from the previous training steps, is below this threshold, more samples are
drawn in that channel, such that on average ``batch_size_threshold`` times the requested number of
samples have a non-zero integrand. Since the number of samples is fixed before sampling, the
channel-wise integral estimates remain unbiased. To limit the cost of channels where almost all
samples are cut away, at most ``max_oversampling`` (default 10) times the requested number of
samples are generated in every channel. Note that the threshold is applied separately to every
channel. In MadNIS 1.x, it was applied to the whole batch, and a full new batch was drawn for all
channels until the threshold was reached.

In addition, the parameter ``drop_zero_integrands``
removes samples with zero-integrand value from the optimization. It is especially useful in
situations where the integrand evaluation is more expensive for samples with non-zero weight.
By default (``rescale_dropped_zeros=True``), the sampling probabilities of the kept samples are
multiplied by the channel-wise ratio of generated to kept samples, as done in MadGraph7. The kept
samples then form an unbiased sample of the region with non-zero integrand, and the channel-wise
means and variances in the loss, the buffered samples and the integration history are consistent
between channels with different cut efficiencies. With ``rescale_dropped_zeros=False``, the kept
samples are used with their original sampling probabilities. Their weights are then too large by
the inverse cut efficiency of their channel, which distorts the relative weight of the channels in
the multi-channel loss.

Invalid integrand values
------------------------

MadNIS checks the values returned by the integrand and raises an exception if they cannot be
integrated. This is the case for non-finite integrand values (nan or inf), for negative or
non-finite prior channel weights, and if all prior channel weights vanish at a point with non-zero
integrand. As in MadGraph7, the integration is aborted, because ignoring these points would bias
the result. The error message contains an example of an affected point. Prior channel weights
that are all zero are allowed at points where the integrand is zero, for example for points that
fail the cuts.

Device and data type
--------------------

The device and data type used for training and sampling can be set using the ``device`` and
``dtype`` arguments of the :py:class:`Integrator <madnis.integrator.Integrator>` constructor.
As the class inherits from ``torch.nn.Module``, the ``to`` function can be used alternatively
to change the device or data type.


Storing and loading trained models
----------------------------------

The :py:class:`Integrator <madnis.integrator.Integrator>` class is a ``torch.nn.Module``. The
functions ``torch.save`` and ``torch.load`` can therefore be used to store and load trained models.
The saved state includes all network parameters and the integration history, but not the buffered
training samples.

.. code-block:: python

    import torch

    # save integrator
    torch.save(integrator.state_dict(), "integrator.pth")
    # load integrator
    integrator.load_state_dict(torch.load("integrator.pth"))
