use super::AnyTensor;
use super::Buffer;
use super::Code;
use super::DataType;
use super::Graph;
use super::Operation;
use super::Result;
use super::SessionOptions;
use super::Status;
use super::Tensor;
use super::TensorType;
use crate::tf;
use libc::{c_char, c_int};
use std::ffi::CStr;
use std::ffi::CString;
use std::marker;
use std::path::Path;
use std::ptr;

/// Aggregation type for a saved model bundle.
#[derive(Debug)]
pub struct SavedModelBundle {
    /// The loaded session.
    pub session: Session,
    /// A meta graph definition as raw protocol buffer.
    pub meta_graph_def: Vec<u8>,
}

impl SavedModelBundle {
    /// Loads a session from an exported model, creating a bundle
    pub fn load<P: AsRef<Path>, Tag: AsRef<str>, Tags: IntoIterator<Item = Tag>>(
        options: &SessionOptions,
        tags: Tags,
        graph: &mut Graph,
        export_dir: P,
    ) -> Result<SavedModelBundle> {
        let mut status = Status::new();

        let export_dir_cstr = export_dir
            .as_ref()
            .to_str()
            .and_then(|s| CString::new(s.as_bytes()).ok())
            .ok_or_else(|| invalid_arg!("Invalid export directory path"))?;

        let tags_cstr: Vec<_> = tags
            .into_iter()
            .map(|t| CString::new(t.as_ref()))
            .collect::<::std::result::Result<_, _>>()
            .map_err(|_| invalid_arg!("Invalid tag name"))?;
        let tags_ptr: Vec<*const c_char> = tags_cstr.iter().map(|t| t.as_ptr()).collect();

        // The empty TF_Buffer will be filled by LoadSessionFromSavedModel
        let mut meta = unsafe { Buffer::<u8>::from_ptr(ptr::null_mut(), 0) };

        let inner = unsafe {
            tf::TF_LoadSessionFromSavedModel(
                options.inner,
                ptr::null(),
                export_dir_cstr.as_ptr(),
                tags_ptr.as_ptr(),
                tags_ptr.len() as c_int,
                graph.inner(),
                meta.inner_mut(),
                status.inner(),
            )
        };
        if inner.is_null() {
            Err(status)
        } else {
            let session = Session { inner: inner };
            Ok(SavedModelBundle {
                session: session,
                meta_graph_def: Vec::from(meta.as_ref()),
            })
        }
    }
}

/// Manages a single graph and execution.
#[derive(Debug)]
pub struct Session {
    inner: *mut tf::TF_Session,
}

impl Session {
    /// Creates a session.
    /// `graph` will be be kept alive for the lifetime of the returned session.
    /// New nodes can still be added to `graph` after this call.
    pub fn new(options: &SessionOptions, graph: &Graph) -> Result<Self> {
        let mut status = Status::new();
        let inner = unsafe { tf::TF_NewSession(graph.inner(), options.inner, status.inner()) };
        if inner.is_null() {
            Err(status)
        } else {
            Ok(Session { inner: inner })
        }
    }

    /// Loads a session from an exported model.
    pub fn from_saved_model<P: AsRef<Path>, Tag: AsRef<str>, Tags: IntoIterator<Item = Tag>>(
        options: &SessionOptions,
        tags: Tags,
        graph: &mut Graph,
        export_dir: P,
    ) -> Result<Self> {
        let mut status = Status::new();

        let export_dir_cstr = export_dir
            .as_ref()
            .to_str()
            .and_then(|s| CString::new(s.as_bytes()).ok())
            .ok_or_else(|| invalid_arg!("Invalid export directory path"))?;

        let tags_cstr: Vec<_> = tags
            .into_iter()
            .map(|t| CString::new(t.as_ref()))
            .collect::<::std::result::Result<_, _>>()
            .map_err(|_| invalid_arg!("Invalid tag name"))?;
        // keeping tags_cstr to retain strings in memory
        let tags_ptr: Vec<*const c_char> = tags_cstr.iter().map(|t| t.as_ptr()).collect();

        let inner = unsafe {
            tf::TF_LoadSessionFromSavedModel(
                options.inner,
                ptr::null(),
                export_dir_cstr.as_ptr(),
                tags_ptr.as_ptr(),
                tags_ptr.len() as c_int,
                graph.inner(),
                ptr::null_mut(),
                status.inner(),
            )
        };
        if inner.is_null() {
            Err(status)
        } else {
            Ok(Session { inner: inner })
        }
    }

    /// Closes the session.
    pub fn close(&mut self) -> Result<()> {
        let mut status = Status::new();
        unsafe {
            tf::TF_CloseSession(self.inner, status.inner());
        }
        status.into_result()
    }

    /// Runs the graph, feeding the inputs and then fetching the outputs
    /// requested in the step.  Note that the session has interior mutability;
    /// this may mutate variables in the graph, and the caller is responsible
    /// for handling race conditions.
    pub fn run(&self, step: &mut SessionRunArgs<'_>) -> Result<()> {
        // In case we're running it a second time and not all outputs were taken out.
        step.drop_output_tensors();

        let mut status = Status::new();
        let maybe_tensors: Result<_> = step.input_tensors.iter().map(|t| t.inner()).collect();
        let input_tensors: Vec<_> = maybe_tensors?;
        unsafe {
            tf::TF_SessionRun(
                self.inner,
                ptr::null(),
                step.input_ports.as_ptr(),
                input_tensors.as_ptr() as *const *const tf::TF_Tensor,
                input_tensors.len() as c_int,
                step.output_ports.as_ptr(),
                step.output_tensors.as_mut_ptr(),
                step.output_tensors.len() as c_int,
                step.target_operations.as_mut_ptr(),
                step.target_operations.len() as c_int,
                ptr::null_mut(),
                status.inner(),
            );
        };
        status.into_result()
    }

    /// Lists all devices in a session.
    pub fn device_list(&self) -> Result<Vec<Device>> {
        let status = Status::new();
        unsafe {
            let list = tf::TF_SessionListDevices(self.inner, status.inner);
            if !status.is_ok() {
                return Err(status);
            }
            let result = (|| {
                let n = tf::TF_DeviceListCount(list);
                let mut devices = Vec::with_capacity(n as usize);
                for i in 0..n {
                    let c_name = tf::TF_DeviceListName(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    let c_type = tf::TF_DeviceListType(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    let bytes = tf::TF_DeviceListMemoryBytes(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    let incarnation = tf::TF_DeviceListIncarnation(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    devices.push(Device {
                        name: CStr::from_ptr(c_name).to_str()?.to_string(),
                        device_type: CStr::from_ptr(c_type).to_str()?.to_string(),
                        memory_bytes: bytes,
                        incarnation,
                    });
                }
                Ok(devices)
            })();
            tf::TF_DeleteDeviceList(list);
            result
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let mut status = Status::new();
        unsafe {
            tf::TF_DeleteSession(self.inner, status.inner());
        }
        // TODO: What do we do with the status?
    }
}

unsafe impl Send for Session {}

unsafe impl Sync for Session {}

////////////////////////

/// An opaque token for retrieving an output from a computation.
#[derive(Copy, Clone, Debug)]
pub struct FetchToken {
    index: usize,
}

/// Deprecated alias for FetchToken.
#[deprecated(note = "Use FetchToken instead.", since = "0.10.0")]
pub type OutputToken = FetchToken;

/// Manages the inputs and outputs for a single execution of a graph.
///
/// Typical usage involves creating an instance of this struct,
/// adding some inputs to it, requesting some outputs, passing it to `Session::run`
/// and then taking the outputs out of it.
///
/// Example:
///
/// ```rust,ignore
/// let mut args = SessionRunArgs::new();
/// args.add_feed(&op1, 0, &tensor1);
/// args.add_feed(&op2, 0, &tensor2);
/// let result_token = args.request_fetch(&op3, 0);
/// session.run(&mut args)?;
/// let result_tensor = args.fetch(result_token)?;
/// ```
///
/// See examples/addition.rs for a more concrete example.
#[derive(Debug)]
pub struct SessionRunArgs<'l> {
    input_ports: Vec<tf::TF_Output>,
    input_tensors: Vec<&'l dyn AnyTensor>,

    output_ports: Vec<tf::TF_Output>,
    output_tensors: Vec<*mut tf::TF_Tensor>,

    target_operations: Vec<*const tf::TF_Operation>,

    phantom: marker::PhantomData<&'l ()>,
}

impl<'l> SessionRunArgs<'l> {
    /// Creates a SessionRunArgs.
    pub fn new() -> Self {
        SessionRunArgs {
            input_ports: vec![],
            input_tensors: vec![],

            output_ports: vec![],
            output_tensors: vec![],

            target_operations: vec![],

            phantom: marker::PhantomData,
        }
    }

    /// Adds an input to be fed to the graph. The index selects which output of
    /// the operation to feed. For most operations, there is only one output,
    /// so the index should be 0.
    pub fn add_feed<T: TensorType>(
        &mut self,
        operation: &Operation,
        index: c_int,
        tensor: &'l Tensor<T>,
    ) {
        self.input_ports.push(tf::TF_Output {
            oper: operation.inner(),
            index: index,
        });
        self.input_tensors.push(tensor);
    }

    /// Deprecated alias for add_feed.
    #[deprecated(note = "Use add_feed instead.", since = "0.10.0")]
    pub fn add_input<T: TensorType>(
        &mut self,
        operation: &Operation,
        index: c_int,
        tensor: &'l Tensor<T>,
    ) {
        self.add_feed(operation, index, tensor)
    }

    /// Requests that an output is fetched from the graph after running this
    /// step. The index selects which output of the operation to return. For
    /// most operations, there is only one output, so the index should be 0.
    /// Returns a token that you can then use to fetch this output from the args
    /// after running it.
    pub fn request_fetch(&mut self, operation: &Operation, index: c_int) -> FetchToken {
        self.output_ports.push(tf::TF_Output {
            oper: operation.inner(),
            index: index,
        });
        self.output_tensors.push(ptr::null_mut());
        FetchToken {
            index: self.output_tensors.len() - 1,
        }
    }

    /// Deprecated alias for request_fetch.
    #[deprecated(note = "Use request_fetch instead.", since = "0.10.0")]
    #[allow(deprecated)]
    pub fn request_output(&mut self, operation: &Operation, index: c_int) -> OutputToken {
        self.request_fetch(operation, index)
    }

    /// Extracts a tensor output given a token. A given token can only be
    /// extracted once per `Session::run`. Returns an error if the token is
    /// invalid, output is unavailable or the requested type does not match the
    /// type of the actual tensor.
    pub fn fetch<T: TensorType>(&mut self, token: FetchToken) -> Result<Tensor<T>> {
        let output_idx = token.index;
        if output_idx >= self.output_tensors.len() {
            return Err(Status::new_set(
                Code::OutOfRange,
                &format!(
                    "Requested output index is out of range: {} vs \
                     {}",
                    output_idx,
                    self.output_tensors.len()
                ),
            )
            .unwrap());
        }
        if self.output_tensors[output_idx].is_null() {
            return Err(Status::new_set(
                Code::Unavailable,
                "Output not available. Either it was already taken, or \
                 this step has not been sucessfully run yet.",
            )
            .unwrap());
        }
        let actual_data_type = self.output_data_type(output_idx).unwrap();
        if actual_data_type != T::data_type() {
            return Err(invalid_arg!(
                "Requested tensor type does not match actual tensor type: \
                 {} vs {}",
                actual_data_type,
                T::data_type()
            ));
        }
        let tensor = unsafe { Tensor::from_tf_tensor(self.output_tensors[output_idx]).unwrap() };
        self.output_tensors[output_idx] = ptr::null_mut();
        Ok(tensor)
    }

    /// Deprecated alias for fetch.
    #[deprecated(note = "Use fetch instead.", since = "0.10.0")]
    #[allow(deprecated)]
    pub fn take_output<T: TensorType>(&mut self, token: OutputToken) -> Result<Tensor<T>> {
        self.fetch(token)
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
        unsafe {
            Some(DataType::from_c(tf::TF_TensorType(
                self.output_tensors[output_idx],
            )))
        }
    }

    fn drop_output_tensors(&mut self) {
        for tensor in &mut self.output_tensors {
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

impl<'l> Drop for SessionRunArgs<'l> {
    fn drop(&mut self) {
        self.drop_output_tensors();
    }
}

/// Deprecated alias for SessionRunArgs.
#[deprecated(note = "Use SessionRunArgs instead.", since = "0.10.0")]
pub type StepWithGraph<'l> = SessionRunArgs<'l>;

////////////////////////

/// Metadata about a device.
#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct Device {
    /// Full name of the device (e.g. /job:worker/replica:0/...)
    pub name: String,

    /// Type of device.
    pub device_type: String,

    /// Amount of memory on the device.
    pub memory_bytes: i64,

    /// Incarnation number of the device.
    pub incarnation: u64,
}

////////////////////////

#[cfg(test)]
mod tests {
    use super::super::DataType;
    use super::super::Graph;
    use super::super::Operation;
    use super::super::SessionOptions;
    use super::super::Shape;
    use super::super::Tensor;
    use super::*;

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
            nd.add_input(two);
            nd.add_input(x.clone());
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
        let (session, x_operation, y_operation) = create_session();
        let mut x = <Tensor<f32>>::new(&[2]);
        x[0] = 2.0;
        x[1] = 3.0;
        let mut step = SessionRunArgs::new();
        step.add_feed(&x_operation, 0, &x);
        let output_token = step.request_fetch(&y_operation, 0);
        session.run(&mut step).unwrap();
        let output_tensor = step.fetch::<f32>(output_token).unwrap();
        assert_eq!(output_tensor.len(), 2);
        assert_eq!(output_tensor[0], 4.0);
        assert_eq!(output_tensor[1], 6.0);
    }

    #[test]
    fn test_savedmodelbundle() {
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            &["train", "serve"],
            &mut graph,
            "test_resources/regression-model",
        )
        .unwrap();

        let x_op = graph.operation_by_name_required("x").unwrap();
        let y_op = graph.operation_by_name_required("y").unwrap();
        let y_hat_op = graph.operation_by_name_required("y_hat").unwrap();
        let _train_op = graph.operation_by_name_required("train").unwrap();

        let SavedModelBundle {
            session,
            meta_graph_def,
        } = bundle;

        assert!(!meta_graph_def.is_empty());

        let mut x = <Tensor<f32>>::new(&[1]);
        x[0] = 2.0;
        let mut y = <Tensor<f32>>::new(&[1]);
        y[0] = 4.0;
        let mut step = SessionRunArgs::new();
        step.add_feed(&x_op, 0, &x);
        step.add_feed(&y_op, 0, &y);
        let output_token = step.request_fetch(&y_hat_op, 0);
        session.run(&mut step).unwrap();
        let output_tensor = step.fetch::<f32>(output_token).unwrap();
        assert_eq!(output_tensor.len(), 1);
    }

    #[test]
    fn test_device_list() {
        let (session, _, _) = create_session();
        let devices = session.device_list().unwrap();
        assert!(
            devices.iter().any(|d| d.device_type == "CPU"),
            "devices: {:?}",
            devices
        );
    }
}
