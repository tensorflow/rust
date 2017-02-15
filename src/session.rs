extern crate tensorflow_sys as tf;

use libc::c_int;
use std::marker;
use std::ptr;
use super::Code;
use super::DataType;
use super::Graph;
use super::GraphTrait;
use super::Operation;
use super::OperationTrait;
use super::Result;
use super::SessionOptions;
use super::Status;
use super::Tensor;
use super::TensorType;

/// Manages a single graph and execution.
#[derive(Debug)]
pub struct Session {
    inner: *mut tf::TF_Session,
}

#[deprecated(note = "Use Session instead.")]
type SessionWithGraph = Session;

impl Session {
    /// Creates a session.
    pub fn new(options: &SessionOptions, graph: &Graph) -> Result<Self> {
        let status = Status::new();
        let inner = unsafe { tf::TF_NewSession(graph.inner(), options.inner, status.inner) };
        if inner.is_null() {
            Err(status)
        } else {
            Ok(Session { inner: inner })
        }
    }

    /// Closes the session.
    pub fn close(&mut self) -> Result<()> {
        let status = Status::new();
        unsafe {
            tf::TF_CloseSession(self.inner, status.inner);
        }
        status.as_result()
    }

    /// Runs the graph, feeding the inputs and then fetching the outputs requested in the step.
    pub fn run(&mut self, step: &mut StepWithGraph) -> Result<()> {
        // Copy the input tensors because TF_Run consumes them.
        let mut input_tensors = Vec::with_capacity(step.input_tensors.len());
        for &input_tensor in &step.input_tensors {
            unsafe {
                let mut dims = Vec::with_capacity(tf::TF_NumDims(input_tensor) as usize);
                for i in 0..dims.capacity() {
                    dims.push(tf::TF_Dim(input_tensor, i as c_int));
                }
                input_tensors.push(tf::TF_NewTensor(tf::TF_TensorType(input_tensor),
                                                    dims.as_ptr(),
                                                    dims.len() as c_int,
                                                    tf::TF_TensorData(input_tensor),
                                                    tf::TF_TensorByteSize(input_tensor),
                                                    Some(super::noop_deallocator),
                                                    ptr::null_mut()));
            }
        }

        // In case we're running it a second time and not all outputs were taken out.
        step.drop_output_tensors();

        let status = Status::new();
        unsafe {
            tf::TF_SessionRun(self.inner,
                              ptr::null(),
                              step.input_ports.as_ptr(),
                              input_tensors.as_mut_ptr(),
                              input_tensors.len() as c_int,
                              step.output_ports.as_ptr(),
                              step.output_tensors.as_mut_ptr(),
                              step.output_tensors.len() as c_int,
                              step.target_operations.as_mut_ptr(),
                              step.target_operations.len() as c_int,
                              ptr::null_mut(),
                              status.inner);
        };
        status.as_result()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let status = Status::new();
        unsafe {
            tf::TF_DeleteSession(self.inner, status.inner);
        }
        // TODO: What do we do with the status?
    }
}

////////////////////////

/// An opaque token for retrieving an output from a computation.
#[derive(Copy,Clone,Debug)]
pub struct OutputToken {
    index: usize,
}

/// Manages the inputs and outputs for a single execution of a graph.
///
/// Typical usage involves creating an instance of this struct,
/// adding some inputs to it, requesting some outputs, passing it to `Session::run`
/// and then taking the outputs out of it.
///
/// This will be renamed to Step once the old API goes away.
#[derive(Debug)]
pub struct StepWithGraph<'l> {
    input_ports: Vec<tf::TF_Output>,
    input_tensors: Vec<*mut tf::TF_Tensor>,

    output_ports: Vec<tf::TF_Output>,
    output_tensors: Vec<*mut tf::TF_Tensor>,

    target_operations: Vec<*const tf::TF_Operation>,

    phantom: marker::PhantomData<&'l ()>,
}

impl<'l> StepWithGraph<'l> {
    /// Creates a StepWithGraph.
    pub fn new() -> Self {
        StepWithGraph {
            input_ports: vec![],
            input_tensors: vec![],

            output_ports: vec![],
            output_tensors: vec![],

            target_operations: vec![],

            phantom: marker::PhantomData,
        }
    }

    /// Adds an input to be fed to the graph.
    pub fn add_input<T: TensorType>(&mut self,
                                    operation: &Operation,
                                    index: c_int,
                                    tensor: &'l Tensor<T>) {
        self.input_ports.push(tf::TF_Output {
            operation: operation.inner(),
            index: index,
        });
        self.input_tensors.push(tensor.inner);
    }

    /// Requests that an output is fetched from the graph after running this step.
    /// Returns an index that you can then use to fetch this output from the step after running it.
    pub fn request_output(&mut self, operation: &Operation, index: c_int) -> OutputToken {
        self.output_ports.push(tf::TF_Output {
            operation: operation.inner(),
            index: index,
        });
        self.output_tensors.push(ptr::null_mut());
        OutputToken { index: self.output_tensors.len() - 1 }
    }

    /// Extracts a tensor output given an index. A given index can only be extracted
    /// once per `Session::run`.
    /// Returns an error if output_idx is out of range, output is unavailable or the
    /// requested type does not match the type of the actual tensor.
    pub fn take_output<T: TensorType>(&mut self, token: OutputToken) -> Result<Tensor<T>> {
        let output_idx = token.index;
        if output_idx >= self.output_tensors.len() {
            return Err(Status::new_set(Code::OutOfRange,
                                       &format!("Requested output index is out of range: {} vs \
                                                 {}",
                                                output_idx,
                                                self.output_tensors.len()))
                .unwrap());
        }
        if self.output_tensors[output_idx].is_null() {
            return Err(Status::new_set(Code::Unavailable,
                                       "Output not available. Either it was already taken, or \
                                        this step has not been sucessfully run yet.")
                .unwrap());
        }
        let actual_data_type = self.output_data_type(output_idx).unwrap();
        if actual_data_type != T::data_type() {
            return Err(invalid_arg!("Requested tensor type does not match actual tensor type: \
                                     {} vs {}",
                                    actual_data_type,
                                    T::data_type()));
        }
        let tensor = unsafe { Tensor::from_tf_tensor(self.output_tensors[output_idx]).unwrap() };
        self.output_tensors[output_idx] = ptr::null_mut();
        Ok(tensor)
    }

    /// Adds a target operation to be executed when running the graph.
    pub fn add_target(&mut self, operation: &Operation) {
        self.target_operations.push(operation.inner());
    }

    /// Retuns the type of the tensor given an index.
    /// Returns `None` if the index is out of range or the output is not yet available.
    pub fn output_data_type(&self, output_idx: usize) -> Option<DataType> {
        if output_idx >= self.output_tensors.len() {
            return None;
        }
        if self.output_tensors[output_idx].is_null() {
            return None;
        }
        unsafe { Some(DataType::from_c(tf::TF_TensorType(self.output_tensors[output_idx]))) }
    }

    fn drop_output_tensors(&mut self) {
        for mut tensor in &mut self.output_tensors {
            // TODO: Is TF_DeleteTensor NULL safe?
            if !tensor.is_null() {
                unsafe {
                    tf::TF_DeleteTensor(*tensor);
                }
            }
            *tensor = ptr::null_mut();
        }
    }
}

impl<'l> Drop for StepWithGraph<'l> {
    fn drop(&mut self) {
        self.drop_output_tensors();
    }
}

////////////////////////

#[cfg(test)]
mod tests {
    extern crate tensorflow_sys as tf;
    use super::*;
    use super::super::DataType;
    use super::super::Graph;
    use super::super::Operation;
    use super::super::Output;
    use super::super::SessionOptions;
    use super::super::Shape;
    use super::super::Tensor;

    fn create_session() -> (Session, Operation, Operation) {
        let mut g = Graph::new();
        let two = {
            let mut nd = g.new_operation("Const", "two").unwrap();
            nd.set_attr_type("dtype", DataType::Float).unwrap();
            let mut value = Tensor::new(&[1]);
            value[0] = 2.0f32;
            nd.set_attr_tensor("value", value).unwrap();
            nd.finish().unwrap()
        };
        let x = {
            let mut nd = g.new_operation("Placeholder", "x").unwrap();
            nd.set_attr_type("dtype", DataType::Float).unwrap();
            nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
            nd.finish().unwrap()
        };
        let y = {
            let mut nd = g.new_operation("Mul", "y").unwrap();
            nd.add_input(Output {
                operation: &two,
                index: 0,
            });
            nd.add_input(Output {
                operation: &x,
                index: 0,
            });
            nd.finish().unwrap()
        };
        let options = SessionOptions::new();
        match Session::new(&options, &g) {
            Ok(session) => (session, x, y),
            Err(status) => panic!("Creating session failed with status: {}", status),
        }
    }

    #[test]
    fn smoke() {
        create_session();
    }

    #[test]
    fn test_close() {
        let (mut session, _, _) = create_session();
        let status = session.close();
        assert!(status.is_ok());
    }

    #[test]
    fn test_run() {
        let (mut session, x_operation, y_operation) = create_session();
        let mut x = <Tensor<f32>>::new(&[2]);
        x[0] = 2.0;
        x[1] = 3.0;
        let mut step = StepWithGraph::new();
        step.add_input(&x_operation, 0, &x);
        let output_token = step.request_output(&y_operation, 0);
        session.run(&mut step).unwrap();
        let output_tensor = step.take_output::<f32>(output_token).unwrap();
        let data = output_tensor.data();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], 4.0);
        assert_eq!(data[1], 6.0);
    }
}
