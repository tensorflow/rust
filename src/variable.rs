use crate::ops;
use crate::AnyTensor;
use crate::DataType;
use crate::Operation;
use crate::Output;
use crate::Result;
use crate::Scope;
use crate::Shape;
use crate::Tensor;
use crate::TensorType;
use std::borrow::Borrow;

/// Holds state in the form of a tensor that persists across steps.
#[derive(Debug, Clone)]
pub struct Variable {
    pub(crate) name: String,
    pub(crate) initializer: Operation,
    pub(crate) output: Output,
    pub(crate) dtype: DataType,
    pub(crate) shape: Shape,
}

impl Variable {
    /// Creates a builder which can be used to create a Variable.
    pub fn builder<'a>() -> VariableBuilder<'a> {
        VariableBuilder::default()
    }

    /// Returns the name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the output which evaluates to the value of the variable.
    pub fn output(&self) -> &Output {
        &self.output
    }

    /// Returns the initializer.
    pub fn initializer(&self) -> &Operation {
        &self.initializer
    }

    /// Returns the data type.
    pub fn data_type(&self) -> DataType {
        self.dtype
    }

    /// Returns the shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[derive(Debug)]
enum VariableInitialValue<'a> {
    Unspecified,
    TensorBox(Box<dyn AnyTensor>),
    TensorRef(&'a dyn AnyTensor),
    Output(Output),
}

/// Builds a Variable.
#[derive(Debug)]
pub struct VariableBuilder<'a> {
    initial_value: VariableInitialValue<'a>,
    shape: Shape,
    dtype: Option<DataType>,
}

impl<'a> Default for VariableBuilder<'a> {
    fn default() -> Self {
        Self {
            initial_value: VariableInitialValue::Unspecified,
            shape: Shape(None),
            dtype: None,
        }
    }
}

impl<'a> VariableBuilder<'a> {
    /// Sets the initial value from anything that can be converted into a Tensor.
    /// This also sets the type and shape.
    pub fn const_initial_value<T: TensorType, TT: Into<Tensor<T>>>(self, value: TT) -> Self {
        let t: Tensor<T> = value.into();
        let shape = t.shape();
        Self {
            initial_value: VariableInitialValue::TensorBox(Box::<Tensor<T>>::new(t)),
            dtype: Some(T::data_type()),
            shape,
        }
    }

    /// Sets the initial value from a Tensor.
    /// This also sets the type and shape.
    pub fn const_initial_tensor<T: TensorType>(self, value: &'a Tensor<T>) -> Self {
        let shape = value.shape();
        Self {
            initial_value: VariableInitialValue::TensorRef(value),
            dtype: Some(T::data_type()),
            shape,
        }
    }

    /// Sets the initial value from an existing output in the graph.
    /// The type and shape are not set and will need to be set manually.
    pub fn initial_value<T: Into<Output>>(self, value: T) -> Self {
        Self {
            initial_value: VariableInitialValue::Output(value.into()),
            ..self
        }
    }

    /// Sets the shape of the variable.
    pub fn shape<S: Into<Shape>>(self, shape: S) -> Self {
        Self {
            shape: shape.into(),
            ..self
        }
    }

    /// Sets the data type of the variable.
    pub fn data_type(self, data_type: DataType) -> Self {
        Self {
            dtype: Some(data_type),
            ..self
        }
    }

    /// Builds the Variable.
    pub fn build(self, scope: &mut Scope) -> Result<Variable> {
        let name = scope.get_unique_name_for_op("VariableV2");
        let dtype = match self.dtype {
            Some(d) => d,
            None => return Err(invalid_arg!("data_type must be specified")),
        };
        let variable_op = {
            let mut graph = scope.graph_mut();
            let mut nd = graph.new_operation("VariableV2", &name)?;
            nd.set_attr_type("dtype", dtype)?;
            nd.set_attr_shape("shape", &self.shape)?;
            nd.finish()?
        };
        let initial_value = match self.initial_value {
            VariableInitialValue::Unspecified => {
                return Err(invalid_arg!("an initial value is required"))
            }
            VariableInitialValue::TensorBox(t) => ops::any_constant(t.borrow(), scope)?.into(),
            VariableInitialValue::TensorRef(t) => ops::any_constant(t, scope)?.into(),
            VariableInitialValue::Output(o) => o,
        };
        let initializer = ops::assign(variable_op.clone(), initial_value, scope)?;
        Ok(Variable {
            name,
            output: variable_op.into(),
            initializer,
            dtype,
            shape: self.shape,
        })
    }
}

////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Code;
    use crate::Session;
    use crate::SessionOptions;
    use crate::SessionRunArgs;

    #[test]
    fn const_initialized_scalar() {
        let scope = Scope::new_root_scope();

        let variable = Variable::builder()
            .const_initial_value(3.0f32)
            .build(&mut scope.with_op_name("foo"))
            .unwrap();
        assert_eq!(variable.name, "foo");
        assert_eq!(variable.shape, Shape(Some(vec![])));
        assert_eq!(variable.dtype, DataType::Float);
        assert_eq!(
            variable.output.operation.get_attr_shape("shape").unwrap(),
            Shape(Some(vec![]))
        );
        assert_eq!(
            variable.output.operation.get_attr_type("dtype").unwrap(),
            DataType::Float
        );

        let options = SessionOptions::new();
        let session = Session::new(&options, &scope.graph()).unwrap();
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&variable.initializer);
        session.run(&mut run_args).unwrap();

        let mut run_args = SessionRunArgs::new();
        let fetch = run_args.request_fetch(&variable.output.operation, 0);
        session.run(&mut run_args).unwrap();
        let output = run_args.fetch::<f32>(fetch).unwrap();
        assert_eq!(&output[..], &[3.0f32]);
    }

    #[test]
    fn const_initialized_matrix() {
        let scope = Scope::new_root_scope();

        let initial = Tensor::<i32>::new(&[2, 3])
            .with_values(&[1, 2, 3, 4, 5, 6])
            .unwrap();
        let variable = Variable::builder()
            .const_initial_tensor(&initial)
            .build(&mut scope.with_op_name("foo"))
            .unwrap();
        assert_eq!(variable.name, "foo");
        assert_eq!(variable.shape, Shape(Some(vec![Some(2), Some(3)])));
        assert_eq!(variable.dtype, DataType::Int32);
        assert_eq!(
            variable.output.operation.get_attr_shape("shape").unwrap(),
            Shape(Some(vec![Some(2), Some(3)]))
        );
        assert_eq!(
            variable.output.operation.get_attr_type("dtype").unwrap(),
            DataType::Int32
        );

        let options = SessionOptions::new();
        let session = Session::new(&options, &scope.graph()).unwrap();
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&variable.initializer);
        session.run(&mut run_args).unwrap();

        let mut run_args = SessionRunArgs::new();
        let fetch = run_args.request_fetch(&variable.output.operation, 0);
        session.run(&mut run_args).unwrap();
        let output = run_args.fetch::<i32>(fetch).unwrap();
        assert_eq!(&output[..], &initial[..]);
    }

    #[test]
    fn custom_initializer_missing_dtype() {
        let mut scope = Scope::new_root_scope();
        let value = Tensor::new(&[]).with_values(&[3.0f32]).unwrap();
        let const_op = ops::constant(value, &mut scope).unwrap();

        assert_eq!(
            Variable::builder()
                .initial_value(const_op)
                .build(&mut scope.with_op_name("foo"))
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn custom_initializer() {
        let mut scope = Scope::new_root_scope();
        let value = Tensor::new(&[]).with_values(&[3.0f32]).unwrap();
        let const_op = ops::constant(value, &mut scope).unwrap();

        let variable = Variable::builder()
            .initial_value(const_op)
            .data_type(DataType::Float)
            .build(&mut scope.with_op_name("foo"))
            .unwrap();
        assert_eq!(variable.name, "foo");
        assert_eq!(variable.shape, Shape(None));
        assert_eq!(variable.dtype, DataType::Float);
        assert_eq!(
            variable.output.operation.get_attr_shape("shape").unwrap(),
            Shape(None)
        );
        assert_eq!(
            variable.output.operation.get_attr_type("dtype").unwrap(),
            DataType::Float
        );

        let options = SessionOptions::new();
        let session = Session::new(&options, &scope.graph()).unwrap();
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&variable.initializer);
        session.run(&mut run_args).unwrap();

        let mut run_args = SessionRunArgs::new();
        let fetch = run_args.request_fetch(&variable.output.operation, 0);
        session.run(&mut run_args).unwrap();
        let output = run_args.fetch::<f32>(fetch).unwrap();
        assert_eq!(&output[..], &[3.0f32]);
    }
}
