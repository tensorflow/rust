use crate::{ops, Operation, Scope, Session, SessionRunArgs, Status, Tensor, Variable};

#[derive(Debug)]
struct SaveRestoreOps {
    prefix_save: Operation,
    prefix_restore: Operation,
    save_op: Operation,
    restore_op: Operation,
}

/// This struct supports saving and restoring variables using Tensorflow checkpoints in SaveV2 format.
/// First, the user creates a [`CheckpointMaker`] attached to an [`Scope`] indicating the list of variables to be saved/restored.
/// The CheckpointMaker lazily modifies the graph creating the nodes needed for saving/restoring.
/// When one wants to save/restore from or into a session, one calls the save/restore  methods
/// # Example
/// ```
/// let mut scope = Scope::new_root_scope();
/// // add operations to define the graph
/// // ...
/// // let w and b the variables that we wish to save
/// let mut checkpoint_maker = CheckpointMaker::new(scope.new_sub_scope("checkpoint"),
///             vec![w.clone(), b.clone()].into_boxed_slice(),
///         );
/// let session = Session::new(&SessionOptions::new(), &scope.graph())?;
/// // run some training
/// // ...
/// // to save the training
/// checkpoint_maker.save(&session, "data/checkpoint")?;
/// // then we restore in a different session to continue there
/// let new_session = Session::new(&SessionOptions::new(), &scope.graph())?;
/// checkpoint_maker.restore(&new_session, "data/checkpoint")?;
/// ```
///
#[derive(Debug)]
pub struct CheckpointMaker {
    scope: Scope,
    variables: Box<[Variable]>,
    save_restore_ops: Option<SaveRestoreOps>,
}

impl CheckpointMaker {
    /// Creates a new CheckpointMaker for a Scope, with a list of variables to save/restore.
    /// The scope is used to modify the graph to add the save and restore ops.
    ///
    /// In order to provide a scope for the CheckpointMaker one can use scope.new_sub_scope("checkpoint")
    /// in order to create the nodes with scoped names.
    pub fn new(scope: Scope, variables: Box<[Variable]>) -> CheckpointMaker {
        CheckpointMaker {
            scope,
            variables,
            save_restore_ops: None,
        }
    }

    // Add save and restore ops to the graph.
    fn build_save_ops(&mut self) -> Result<SaveRestoreOps, Status> {
        let mut all_variable_ops_opt: Option<Vec<Operation>> = None;

        let existing_save_op = self.scope.graph().operation_by_name("save")?;
        let (prefix_save, save_op) = if let Some(op) = existing_save_op {
            let prefix_save_op = self
                .scope
                .graph()
                .operation_by_name_required("prefix_save")?;
            (prefix_save_op, op)
        } else {
            let all_variable_ops = all_variable_ops_opt.get_or_insert_with(|| {
                self.variables
                    .iter()
                    .map(|v| v.output.operation.clone())
                    .collect::<Vec<_>>()
            });
            let prefix_save = ops::Placeholder::new()
                .dtype(crate::DataType::String)
                .build(&mut self.scope.with_op_name("prefix_save"))?;
            let tensor_names = ops::constant(
                self.variables
                    .iter()
                    .map(|v| String::from(v.name()))
                    .collect::<Vec<_>>()
                    .as_slice(),
                &mut self.scope,
            )?;
            let shape_and_slices = ops::constant(
                &self
                    .variables
                    .iter()
                    .map(|_| "".to_string())
                    .collect::<Vec<_>>()[..],
                &mut self.scope,
            )?;
            let tensors = all_variable_ops
                .iter()
                .map(|v| v.output(0))
                .collect::<Vec<_>>();

            let mut g = self.scope.graph_mut();
            let mut nd = g.new_operation("SaveV2", "save")?;
            nd.add_input(prefix_save.clone());
            nd.add_input(tensor_names);
            nd.add_input(shape_and_slices);
            nd.add_input_list(&tensors[..]);

            let dtypes = all_variable_ops
                .iter()
                .map(|v| v.get_attr_type("dtype"))
                .collect::<Result<Vec<_>, Status>>()?;
            nd.set_attr_type_list("dtypes", &dtypes[..])?;
            let save_op = nd.finish()?;
            (prefix_save, save_op)
        };
        let opt_restore_op = self.scope.graph().operation_by_name("restore")?;
        let (prefix_restore, restore_op) = if let Some(op) = opt_restore_op {
            let the_prefix_restore = self
                .scope
                .graph()
                .operation_by_name_required("prefix_restore")?;
            (the_prefix_restore, op)
        } else {
            let all_variable_ops = all_variable_ops_opt.get_or_insert_with(|| {
                self.variables
                    .iter()
                    .map(|v| v.output.operation.clone())
                    .collect::<Vec<_>>()
            });
            let prefix_restore = ops::Placeholder::new()
                .dtype(crate::DataType::String)
                .build(&mut self.scope.with_op_name("prefix_restore"))?;
            let all_var_names = self
                .variables
                .iter()
                .map(|v| v.name.clone())
                .collect::<Vec<_>>();
            let tensor_names = ops::constant(&all_var_names[..], &mut self.scope)?;
            let shape_and_slices = ops::constant(
                &self
                    .variables
                    .iter()
                    .map(|_| "".to_string())
                    .collect::<Vec<_>>()[..],
                &mut self.scope,
            )?;
            let mut g = self.scope.graph_mut();
            let mut nd = g.new_operation("RestoreV2", "restore")?;
            nd.add_input(prefix_restore.clone());
            nd.add_input(tensor_names);
            nd.add_input(shape_and_slices);
            let dtypes = all_variable_ops
                .iter()
                .map(|v| v.get_attr_type("dtype"))
                .collect::<Result<Vec<_>, Status>>()?;
            nd.set_attr_type_list("dtypes", &dtypes[..])?;
            let restore_op = nd.finish()?;
            drop(g);
            let mut restore_var_ops = Vec::<Operation>::new();
            for (i, var) in self.variables.iter().enumerate() {
                let var_op = var.output.operation.clone();
                restore_var_ops.push(ops::assign(
                    var_op,
                    crate::Output {
                        operation: restore_op.clone(),
                        index: i as i32,
                    },
                    &mut self.scope.new_sub_scope(format!("restore{}", i).as_str()),
                )?);
            }
            let mut no_op = ops::NoOp::new();
            for op in restore_var_ops {
                no_op = no_op.add_control_input(op);
            }
            (prefix_restore, no_op.build(&mut self.scope)?)
        };
        Ok(SaveRestoreOps {
            prefix_save,
            prefix_restore,
            save_op,
            restore_op,
        })
    }

    fn get_save_operation(&mut self) -> Result<&SaveRestoreOps, Status> {
        if self.save_restore_ops.is_none() {
            self.save_restore_ops = Some(self.build_save_ops()?);
        }
        let save_r_op_ref = self.save_restore_ops.as_ref();
        // SAFETY: the condition above has ensured that self.save_restore_ops is Some(_)
        let save_r_op = unsafe { save_r_op_ref.unwrap_unchecked() };
        Ok(save_r_op)
    }

    /// Save the variables listed in this CheckpointMaker to the checkpoint with base filename backup_filename_base.
    pub fn save(&mut self, session: &Session, backup_filename_base: &str) -> Result<(), Status> {
        let save_restore_ops = self.get_save_operation()?;
        let prefix_arg = Tensor::from(backup_filename_base.to_string());
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&save_restore_ops.prefix_save, 0, &prefix_arg);
        run_args.add_target(&save_restore_ops.save_op);
        session.run(&mut run_args)?;
        Ok(())
    }

    /// Restore into the session the variables listed in this CheckpointMaker from the checkpoint
    /// in path_base.
    pub fn restore(&mut self, session: &Session, path_base: &str) -> Result<(), Status> {
        let save_restore_ops = self.get_save_operation()?;
        let prefix_arg = Tensor::from(path_base.to_string());
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&save_restore_ops.prefix_restore, 0, &prefix_arg);
        run_args.add_target(&save_restore_ops.restore_op);
        session.run(&mut run_args)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::Placeholder;
    use crate::{
        ops, CheckpointMaker, Code, DataType, FetchToken, Operation, Scope, Session,
        SessionOptions, SessionRunArgs, Status, Tensor, Variable,
    };

    fn make_variable(
        scope: &mut Scope,
        name: &str,
        dims: &[u64],
        values: &[f32],
    ) -> Result<Variable, Status> {
        Ok(Variable::builder()
            .const_initial_value(Tensor::new(dims).with_values(values)?)
            .data_type(DataType::Float)
            .build(&mut scope.with_op_name(name))?)
    }

    fn create_assignment(
        var: &Variable,
        scope: &mut Scope,
    ) -> Result<(Operation, Operation), Status> {
        let placeholder = Placeholder::new()
            .dtype(DataType::Float)
            .shape(var.shape.clone())
            .build(&mut scope.with_op_name(var.name.as_str()))?;
        Ok((
            placeholder.clone(),
            ops::assign(var.output.clone(), placeholder, scope)?,
        ))
    }

    struct MyScopeData {
        scope: Scope,
        variables: [Variable; 3],
    }

    // Initialize a scope and place same variables in it
    fn create_scope() -> Result<MyScopeData, Status> {
        let mut scope = Scope::new_root_scope();
        let var_w = make_variable(&mut scope, "w", &[], &[2.2])?;
        let var_b = make_variable(&mut scope, "b", &[3], &[1.0, 2.0, 4.5])?;
        let var_a = make_variable(&mut scope, "a", &[3, 2], &[1.0, 2.0, 3.3, 7.0, 8.0, 8.5])?;
        Ok(MyScopeData {
            scope,
            variables: [var_w, var_b, var_a],
        })
    }

    struct AssignData {
        pub placeholder_ops: Box<[Operation]>,
        pub assign_op: Operation,
    }
    fn add_assign_op(scope_data: &mut MyScopeData) -> Result<AssignData, Status> {
        let mut placeholder_scope = scope_data.scope.new_sub_scope("placeholder");
        let mut placeholders: Vec<Operation> = Vec::new();
        let mut no_op_bld = ops::NoOp::new();
        for var in scope_data.variables.as_ref() {
            let (placeholder, assign_op) = create_assignment(&var, &mut placeholder_scope)?;
            placeholders.push(placeholder);
            no_op_bld = no_op_bld.add_control_input(assign_op);
        }
        let assign_op = no_op_bld.build(&mut scope_data.scope)?;
        Ok(AssignData {
            placeholder_ops: placeholders.into_boxed_slice(),
            assign_op,
        })
    }

    fn assign_variables(
        session: &Session,
        scope_data: &MyScopeData,
        assign_data: &AssignData,
        values: &[&[f32]],
    ) -> Result<(), Status> {
        let mut values_fed: Vec<Tensor<f32>> =
            Vec::with_capacity(assign_data.placeholder_ops.len());
        let mut session_run = SessionRunArgs::new();
        for i_var in 0..assign_data.placeholder_ops.len() {
            let value_fed_as_tensor = Tensor::new(
                &scope_data.variables[i_var]
                    .shape()
                    .0
                    .as_ref()
                    .ok_or(Status::new_set(Code::Internal, "Shape not present")?)?
                    .iter()
                    .map(|o| {
                        o.map(|i| i as u64)
                            .ok_or(Status::new_set(Code::Internal, "Shape item not present")?)
                    })
                    .collect::<Result<Vec<u64>, Status>>()?
                    .as_ref(),
            )
            .with_values(&values[i_var])?;
            values_fed.push(value_fed_as_tensor);
        }
        for i_var in 0..assign_data.placeholder_ops.len() {
            session_run.add_feed(&assign_data.placeholder_ops[i_var], 0, &values_fed[i_var]);
        }
        session_run.add_target(&assign_data.assign_op);
        session.run(&mut session_run)?;
        Ok(())
    }

    fn check_variables(
        session: &Session,
        variables: &[Variable],
        values: &[&[f32]],
    ) -> Result<(), Status> {
        let mut session_run = SessionRunArgs::new();
        let mut tokens: Vec<FetchToken> = Vec::with_capacity(variables.len());
        for i in 0..variables.len() {
            tokens.push(session_run.request_fetch(
                &variables[i].output().operation,
                variables[i].output().index,
            ));
        }
        session.run(&mut session_run)?;
        for i in 0..variables.len() {
            let got_tensor: Tensor<f32> = session_run.fetch(tokens[i])?;
            assert_eq!(values[i], got_tensor.as_ref());
        }
        Ok(())
    }

    #[test]
    fn simple_save() -> Result<(), Box<dyn std::error::Error>> {
        let mut first_scope_data = create_scope()?;
        let assign_data = add_assign_op(&mut first_scope_data)?;
        let first_session = Session::new(&SessionOptions::new(), &first_scope_data.scope.graph())?;
        let new_values: [&[f32]; 3] = [
            &[5.1],
            &[4.0, 2.2, 6.0],
            &[11.0, 12.0, 13.6, 17.1, 18.4, 19.5],
        ];
        assign_variables(&first_session, &first_scope_data, &assign_data, &new_values)?;
        let mut checkpoint = CheckpointMaker::new(
            first_scope_data.scope.new_sub_scope("checkpoint"),
            Box::from(first_scope_data.variables.clone()),
        );
        let temp_dir = tempfile::tempdir()?;
        let checkpoint_path = temp_dir.path().join("checkpoint-vars");
        let checkpoint_path_str = checkpoint_path
            .into_os_string()
            .into_string()
            .map_err(|_| "Cannot convert checkpoint path")?;
        checkpoint.save(&first_session, checkpoint_path_str.as_str())?;
        let MyScopeData {
            scope: second_scope,
            variables: second_variables,
        } = create_scope()?;
        let second_session = Session::new(&SessionOptions::new(), &second_scope.graph())?;
        let mut second_checkpoint =
            CheckpointMaker::new(second_scope, Box::new(second_variables.clone()));
        second_checkpoint.restore(&second_session, checkpoint_path_str.as_str())?;
        check_variables(&second_session, &second_variables, &new_values)?;
        Ok(())
    }
}
