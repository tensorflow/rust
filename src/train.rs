//! This module supports building and training models.

use crate::ops;
use crate::DataType;
use crate::Operation;
use crate::Output;
use crate::Result;
use crate::Scope;
use crate::Tensor;
use crate::TensorType;
use crate::Variable;

/// Options for `Optimizer::minimize`.
#[derive(Default, Debug, Clone)]
pub struct MinimizeOptions<'a> {
    variables: &'a [Variable],
}

impl<'a> MinimizeOptions<'a> {
    /// Sets the variables which will be optimized.
    pub fn with_variables(self, variables: &'a [Variable]) -> Self {
        Self { variables }
    }
}

/// Options for `Optimizer::compute_gradients`.
#[derive(Default, Debug, Clone)]
pub struct ComputeGradientsOptions<'a> {
    variables: &'a [Variable],
}

impl<'a> ComputeGradientsOptions<'a> {
    /// Sets the variables whose gradients need to be computed.
    pub fn with_variables(self, variables: &'a [Variable]) -> Self {
        Self { variables }
    }
}

/// Options for `Optimizer::apply_gradients`.
#[derive(Default, Debug, Clone)]
pub struct ApplyGradientsOptions<'a> {
    grads_and_vars: &'a [(Option<Output>, Variable)],
}

impl<'a> ApplyGradientsOptions<'a> {
    /// Sets the variables which will be optimized and their associated gradients.
    pub fn with_grads_and_vars(self, grads_and_vars: &'a [(Option<Output>, Variable)]) -> Self {
        Self { grads_and_vars }
    }
}

/// An optimizer adjusts variables to minimize some specified value.
///
/// Basic usage only requires calling `minimize`, which calls
/// `compute_gradients` and `apply_gradients` internally.  Advanced users may
/// want to call `compute_gradients` and `apply_gradients` manually to allow
/// them to modify the gradients, e.g. for clipping.
pub trait Optimizer {
    /// Computes the gradient of a value with respect to the given variables.
    /// This adds nodes to the graph, so reuse its results if possible.
    /// Any variable whose gradient cannot be calculated will have a None gradient.
    ///
    /// Users are encouraged to call `minimize` instead unless they need to
    /// manually modify gradients.
    fn compute_gradients(
        &self,
        scope: &mut Scope,
        loss: Output,
        opts: ComputeGradientsOptions,
    ) -> Result<Vec<(Option<Output>, Variable)>> {
        let variable_outputs: Vec<_> = opts.variables.iter().map(|v| v.output.clone()).collect();
        let gradients = scope
            .graph_mut()
            .add_gradients(None, &[loss], &variable_outputs, None)?;
        let mut output = Vec::with_capacity(opts.variables.len());
        for (i, gradient) in gradients.into_iter().enumerate() {
            output.push((gradient, opts.variables[i].clone()));
        }
        Ok(output)
    }

    /// Applies the given gradients to the variables.
    ///
    /// This returns newly created variables which may be needed to track the
    /// optimizer's internal state, as well as an operation which applies the
    /// gradients once.
    ///
    /// Users are encouraged to call `minimize` instead unless they need to
    /// manually modify gradients.
    fn apply_gradients(
        &self,
        scope: &mut Scope,
        opts: ApplyGradientsOptions,
    ) -> Result<(Vec<Variable>, Operation)>;

    /// Adds operations to the graph to minimize loss with respect to the
    /// variables.
    ///
    /// This returns newly created variables which may be needed to track the
    /// optimizers internal state, as well as an operation which performs a
    /// single step of minimization.        
    fn minimize(
        &self,
        scope: &mut Scope,
        loss: Output,
        opts: MinimizeOptions,
    ) -> Result<(Vec<Variable>, Operation)> {
        let grads_and_vars = self.compute_gradients(
            scope,
            loss,
            ComputeGradientsOptions {
                variables: opts.variables,
            },
        )?;
        self.apply_gradients(
            scope,
            ApplyGradientsOptions {
                grads_and_vars: &grads_and_vars,
            },
        )
    }
}

/// Optimizer that implements the gradient descent algorithm.
#[derive(Debug)]
pub struct GradientDescentOptimizer {
    learning_rate: Output,
}

impl GradientDescentOptimizer {
    /// Creates a new optimizer with the given learning rate.
    pub fn new<T: Into<Output>>(learning_rate: T) -> Self {
        Self {
            learning_rate: learning_rate.into(),
        }
    }
}

impl Optimizer for GradientDescentOptimizer {
    fn apply_gradients(
        &self,
        scope: &mut Scope,
        opts: ApplyGradientsOptions,
    ) -> Result<(Vec<Variable>, Operation)> {
        let mut apply_ops = Vec::new();
        for (grad, var) in opts.grads_and_vars {
            if let Some(grad) = grad {
                apply_ops.push(ops::apply_gradient_descent(
                    var.output.clone(),
                    self.learning_rate.clone(),
                    grad.clone(),
                    scope,
                )?);
            }
        }
        let mut nop = ops::NoOp::new();
        for apply_op in &apply_ops {
            nop = nop.add_control_input(apply_op.clone());
        }
        Ok((Vec::new(), nop.build(scope)?))
    }
}

/// Optimizer that implements the Adadelta algorithm.
///
/// See [M. D. Zeiler](https://arxiv.org/abs/1212.5701).
#[derive(Debug)]
pub struct AdadeltaOptimizer {
    learning_rate: Option<Output>,
    rho: Option<Output>,
    epsilon: Option<Output>,
}

impl Default for AdadeltaOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdadeltaOptimizer {
    /// Creates a new optimizer with default parameters (learning_rate=0.001, rho=0.95, epsilon=1e-8).
    pub fn new() -> Self {
        Self {
            learning_rate: None,
            rho: None,
            epsilon: None,
        }
    }

    /// Sets the learning rate.  Default is 0.001.
    pub fn set_learning_rate<T: Into<Output>>(&mut self, learning_rate: T) {
        self.learning_rate = Some(learning_rate.into());
    }

    /// Sets rho, the decay rate.  Default is 0.95.
    pub fn set_rho<T: Into<Output>>(&mut self, rho: T) {
        self.rho = Some(rho.into());
    }

    /// Sets epsilon, the conditioning.  Default is 1e-8.
    pub fn set_epsilon<T: Into<Output>>(&mut self, epsilon: T) {
        self.epsilon = Some(epsilon.into());
    }
}

fn or_constant<T: TensorType, TT: Into<Tensor<T>>>(
    scope: &mut Scope,
    value: &Option<Output>,
    default: TT,
) -> Result<Output> {
    match value {
        Some(x) => Ok(x.clone()),
        None => Ok(ops::constant(default, scope)?.into()),
    }
}

fn create_zeros_slot(
    scope: &mut Scope,
    primary: &Variable,
    dtype: Option<DataType>,
) -> Result<Variable> {
    let dtype = dtype.unwrap_or(primary.dtype);
    let zeros = ops::ZerosLike::new()
        .add_control_input(primary.initializer.clone())
        .build(primary.output.clone(), scope)?;
    Variable::builder()
        .initial_value(zeros)
        .shape(primary.shape.clone())
        .data_type(dtype)
        .build(scope)
}

impl Optimizer for AdadeltaOptimizer {
    fn apply_gradients(
        &self,
        scope: &mut Scope,
        opts: ApplyGradientsOptions,
    ) -> Result<(Vec<Variable>, Operation)> {
        let learning_rate = or_constant(scope, &self.learning_rate, 0.001f32)?;
        let rho = or_constant(scope, &self.rho, 0.95f32)?;
        let epsilon = or_constant(scope, &self.epsilon, 1e-8f32)?;
        let mut apply_ops = Vec::new();
        let mut variables = Vec::new();
        for (grad, var) in opts.grads_and_vars {
            if let Some(grad) = grad {
                let mut scope = scope.new_sub_scope(&var.name);
                let accum = create_zeros_slot(&mut scope.new_sub_scope("accum"), var, None)?;
                let accum_update =
                    create_zeros_slot(&mut scope.new_sub_scope("accum_update"), var, None)?;
                apply_ops.push(ops::apply_adadelta(
                    var.output.clone(),
                    accum.output.clone(),
                    accum_update.output.clone(),
                    learning_rate.clone(),
                    rho.clone(),
                    epsilon.clone(),
                    grad.clone(),
                    &mut scope,
                )?);
                variables.push(accum.clone());
                variables.push(accum_update.clone());
            }
        }
        let mut no_op = ops::NoOp::new();
        for apply_op in &apply_ops {
            no_op = no_op.add_control_input(apply_op.clone());
        }
        Ok((variables, no_op.build(scope)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;
    use crate::Scope;
    use crate::Session;
    use crate::SessionOptions;
    use crate::SessionRunArgs;
    use crate::Tensor;

    #[test]
    fn simple_gradient_descent() {
        let mut scope = Scope::new_root_scope();
        let x_var = Variable::builder()
            .const_initial_value::<_, f32>(3.0)
            .build(&mut scope.with_op_name("x"))
            .unwrap();
        let x_squared = ops::mul(x_var.output.clone(), x_var.output.clone(), &mut scope).unwrap();
        let sgd = GradientDescentOptimizer {
            learning_rate: Output {
                operation: ops::constant(0.1f32, &mut scope).unwrap(),
                index: 0,
            },
        };
        let (minimizer_vars, minimize) = sgd
            .minimize(
                &mut scope,
                x_squared.into(),
                MinimizeOptions::default().with_variables(&[x_var.clone()]),
            )
            .unwrap();
        let options = SessionOptions::new();
        let session = Session::new(&options, &scope.graph()).unwrap();

        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&x_var.initializer);
        for var in &minimizer_vars {
            run_args.add_target(&var.initializer);
        }
        session.run(&mut run_args).unwrap();

        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&minimize);
        let x_fetch = run_args.request_fetch(&x_var.output.operation, 0);

        session.run(&mut run_args).unwrap();
        let x_output = run_args.fetch::<f32>(x_fetch).unwrap();
        assert_eq!(x_output.len(), 1);
        assert!(
            x_output[0] >= 2.39 && x_output[0] <= 2.41,
            "x_output[0] = {}",
            x_output[0]
        );

        session.run(&mut run_args).unwrap();
        let x_output = run_args.fetch::<f32>(x_fetch).unwrap();
        assert_eq!(x_output.len(), 1);
        assert!(
            x_output[0] >= 1.91 && x_output[0] <= 1.93,
            "x_output[0] = {}",
            x_output[0]
        );

        session.run(&mut run_args).unwrap();
        let x_output = run_args.fetch::<f32>(x_fetch).unwrap();
        assert_eq!(x_output.len(), 1);
        assert!(
            x_output[0] >= 1.52 && x_output[0] <= 1.54,
            "x_output[0] = {}",
            x_output[0]
        );
    }

    #[test]
    fn simple_adadelta() {
        let mut scope = Scope::new_root_scope();
        let x_var = Variable::builder()
            .const_initial_value(3.0f32)
            .build(&mut scope.with_op_name("x"))
            .unwrap();
        let x_squared = ops::mul(x_var.output.clone(), x_var.output.clone(), &mut scope).unwrap();
        let mut optimizer = AdadeltaOptimizer::new();
        optimizer.set_learning_rate(ops::constant(0.1f32, &mut scope).unwrap());
        let (minimizer_vars, minimize) = optimizer
            .minimize(
                &mut scope,
                x_squared.into(),
                MinimizeOptions::default().with_variables(&[x_var.clone()]),
            )
            .unwrap();
        let options = SessionOptions::new();
        let session = Session::new(&options, &scope.graph()).unwrap();

        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&x_var.initializer);
        for var in &minimizer_vars {
            run_args.add_target(&var.initializer);
        }
        session.run(&mut run_args).unwrap();

        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&minimize);
        let x_fetch = run_args.request_fetch(&x_var.output.operation, 0);

        session.run(&mut run_args).unwrap();
        let x_output = run_args.fetch::<f32>(x_fetch).unwrap();
        assert_eq!(x_output.len(), 1);
        assert!(
            x_output[0] >= 2.99994 && x_output[0] <= 2.99996,
            "x_output[0] = {}",
            x_output[0]
        );

        session.run(&mut run_args).unwrap();
        let x_output = run_args.fetch::<f32>(x_fetch).unwrap();
        assert_eq!(x_output.len(), 1);
        assert!(
            x_output[0] >= 2.99990 && x_output[0] <= 2.99992,
            "x_output[0] = {}",
            x_output[0]
        );

        session.run(&mut run_args).unwrap();
        let x_output = run_args.fetch::<f32>(x_fetch).unwrap();
        assert_eq!(x_output.len(), 1);
        assert!(
            x_output[0] >= 2.99985 && x_output[0] <= 2.99987,
            "x_output[0] = {}",
            x_output[0]
        );
    }

    #[test]
    fn xor_nn() {
        let mut scope = Scope::new_root_scope();
        let scope = &mut scope;
        let hidden_size: u64 = 4;
        let input = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64, 2])
            .build(&mut scope.with_op_name("input"))
            .unwrap();
        let label = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64])
            .build(&mut scope.with_op_name("label"))
            .unwrap();
        let w_shape = ops::constant(&[2, hidden_size as i64][..], scope).unwrap();
        let w_init = ops::RandomStandardNormal::new()
            .dtype(DataType::Float)
            .build(w_shape, scope)
            .unwrap();
        let w = Variable::builder()
            .initial_value(w_init)
            .data_type(DataType::Float)
            .shape([2, hidden_size])
            .build(&mut scope.with_op_name("w"))
            .unwrap();
        let b = Variable::builder()
            .const_initial_value(Tensor::<f32>::new(&[hidden_size]))
            .build(&mut scope.with_op_name("b"))
            .unwrap();
        let layer1a = ops::MatMul::new()
            .build(input.clone(), w.output.clone(), scope)
            .unwrap();
        let layer1b = ops::Add::new()
            .build(layer1a, b.output.clone(), scope)
            .unwrap();
        let layer1 = ops::Tanh::new().build(layer1b, scope).unwrap();
        let w2_shape = ops::constant(&[hidden_size as i64, 1][..], scope).unwrap();
        let w2_init = ops::RandomStandardNormal::new()
            .dtype(DataType::Float)
            .build(w2_shape, scope)
            .unwrap();
        let w2 = Variable::builder()
            .initial_value(w2_init)
            .data_type(DataType::Float)
            .shape([hidden_size, 1])
            .build(&mut scope.with_op_name("w2"))
            .unwrap();
        let b2 = Variable::builder()
            .const_initial_value(Tensor::<f32>::new(&[1]))
            .build(&mut scope.with_op_name("b2"))
            .unwrap();
        let layer2a = ops::mat_mul(layer1, w2.output.clone(), scope).unwrap();
        let layer2b = ops::add(layer2a, b2.output.clone(), scope).unwrap();
        let layer2 = layer2b;
        let error = ops::sub(layer2.clone(), label.clone(), scope).unwrap();
        let error_squared = ops::mul(error.clone(), error, scope).unwrap();
        let sgd = GradientDescentOptimizer {
            learning_rate: Output {
                operation: ops::constant(0.1f32, scope).unwrap(),
                index: 0,
            },
        };
        let variables = vec![w.clone(), b.clone(), w2.clone(), b2.clone()];
        let (minimizer_vars, minimize) = sgd
            .minimize(
                scope,
                error_squared.clone().into(),
                MinimizeOptions::default().with_variables(&variables),
            )
            .unwrap();
        let options = SessionOptions::new();
        let g = scope.graph_mut();
        let session = Session::new(&options, &g).unwrap();

        let mut run_args = SessionRunArgs::new();
        for var in &variables {
            run_args.add_target(&var.initializer);
        }
        for var in &minimizer_vars {
            run_args.add_target(&var.initializer);
        }
        session.run(&mut run_args).unwrap();

        let mut input_tensor = Tensor::<f32>::new(&[1, 2]);
        let mut label_tensor = Tensor::<f32>::new(&[1]);
        let mut train = |i| {
            input_tensor[0] = (i & 1) as f32;
            input_tensor[1] = ((i >> 1) & 1) as f32;
            label_tensor[0] = ((i & 1) ^ ((i >> 1) & 1)) as f32;
            let mut run_args = SessionRunArgs::new();
            run_args.add_target(&minimize);
            let error_squared_fetch = run_args.request_fetch(&error_squared, 0);
            run_args.add_feed(&input, 0, &input_tensor);
            run_args.add_feed(&label, 0, &label_tensor);
            session.run(&mut run_args).unwrap();
            run_args.fetch::<f32>(error_squared_fetch).unwrap()[0]
        };
        for i in 0..1000 {
            train(i);
        }
        for i in 0..4 {
            let error = train(i);
            assert!(error < 0.01, "error = {}", error);
        }
    }
}
