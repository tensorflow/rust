use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::ErrorKind;
use std::io::Write;
use std::path::Path;
use std::result::Result;
use tensorflow_op_codegen::parser;
use tensorflow_op_codegen::protos::OpDef;

#[derive(Clone)]
struct Attr {
    rust_name: String,
    attr_type: String,
    c_name: String,
}
#[derive(Clone)]
struct Output {
    rust_name: String,
    number_attr: Option<String>,
}

#[derive(Clone)]
struct Input {
    rust_name: String,
    number_attr: Option<String>,
}
/// Input and Output shared behaviour
trait Edge {
    fn rust_name(&self) -> &str;
    fn number_attr(&self) -> Option<&str>;
    fn edge_type(&self) -> &str;
}
impl Edge for &Input {
    fn rust_name(&self) -> &str {
        &self.rust_name
    }
    fn number_attr(&self) -> Option<&str> {
        if let Some(ref number_attr) = self.number_attr {
            Some(number_attr)
        } else {
            None
        }
    }
    fn edge_type(&self) -> &str {
        "Input"
    }
}
impl Edge for &Output {
    fn rust_name(&self) -> &str {
        &self.rust_name
    }
    fn number_attr(&self) -> Option<&str> {
        if let Some(ref number_attr) = self.number_attr {
            Some(number_attr)
        } else {
            None
        }
    }
    fn edge_type(&self) -> &str {
        "Output"
    }
}

fn write_set_attr<W: Write>(w: &mut W, attr: &Attr, node_var: &str) -> Result<(), io::Error> {
    let c_name = &attr.c_name;
    let rust_name = &attr.rust_name;
    let setter = match attr.attr_type.as_str() {
        "::std::string::String" => format!("{}.set_attr_string(\"{}\", value)?", node_var, c_name),
        "crate::DataType" => format!("{}.set_attr_type(\"{}\", *value)?", node_var, c_name),
        "bool" => format!("{}.set_attr_bool(\"{}\", *value)?", node_var, c_name),
        "f32" => format!("{}.set_attr_float(\"{}\", *value)?", node_var, c_name),
        "i64" => format!("{}.set_attr_int(\"{}\", *value)?", node_var, c_name),
        "crate::Shape" => format!("{}.set_attr_shape(\"{}\", value)?", node_var, c_name),
        "crate::Tensor" => format!("{}.set_attr_any_tensor(\"{}\", value)?", node_var, c_name),
        "::std::vec::Vec<::std::string::String>" => {
            format!("{}.set_attr_string_list(\"{}\", value)?", node_var, c_name)
        }
        "::std::vec::Vec<f32>" => {
            format!("{}.set_attr_float_list(\"{}\", value)?", node_var, c_name)
        }
        "::std::vec::Vec<i64>" => format!("{}.set_attr_int_list(\"{}\", value)?", node_var, c_name),
        "::std::vec::Vec<crate::DataType>" => {
            format!("{}.set_attr_type_list(\"{}\", value)?", node_var, c_name)
        }
        "::std::vec::Vec<crate::Shape>" => {
            format!("{}.set_attr_shape_list(\"{}\", value)?", node_var, c_name)
        }
        ty => panic!("Unrecognized attribute type for {}: {}", attr.rust_name, ty),
    };
    writeln!(
        w,
        "        if let ::std::option::Option::Some(value) = &self.{} {{",
        rust_name
    )?;
    writeln!(w, "            {};", setter)?;
    writeln!(w, "        }}")?;
    Ok(())
}

fn write_short_fn<W: Write>(
    w: &mut W,
    name: &str,
    fn_name: &str,
    args: &[String],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<_> = args.iter().map(|arg| escaper.escape(arg)).collect();
    write!(w, "/// Shorthand for `{}::new().build(", name)?;
    for arg in &escaped_args {
        write!(w, "{}, ", &arg)?;
    }
    let scope_var = escaper.escape("scope");
    writeln!(w, "{})`.", scope_var)?;
    write!(w, "pub fn {}<", fn_name)?;
    for i in 0..args.len() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "O{}: ::std::convert::Into<crate::Output>", i)?;
    }
    write!(w, ">(")?;
    for (i, arg) in escaped_args.iter().enumerate() {
        write!(w, "{}: O{}, ", arg, i)?;
    }
    writeln!(
        w,
        "{}: &mut crate::Scope) -> crate::Result<crate::Operation> {{",
        scope_var
    )?;
    write!(w, "    {}::new().build(", name)?;
    for arg in escaped_args {
        write!(w, "{}, ", arg)?;
    }
    writeln!(w, "{})", scope_var)?;
    writeln!(w, "}}")?;
    Ok(())
}

fn write_attr_setter<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    writeln!(w)?;
    writeln!(w, "    /// Sets the `{}` attribute.", &attr.c_name)?;
    let rust_name = &attr.rust_name;
    let attr_type = &attr.attr_type;
    let mut value = "value.into()".to_string();
    if attr_type == "crate::Tensor" {
        value = format!(
            "(::std::boxed::Box::new({}) as ::std::boxed::Box<dyn crate::AnyTensor>)",
            value
        );
        writeln!(
            w,
            "    pub fn {}<T: crate::TensorType, ArgType: ::std::convert::Into<crate::Tensor<T>>>(mut self, value: ArgType) -> Self {{",
            rust_name
        )?;
    } else {
        writeln!(
            w,
            "    pub fn {}<ArgType: ::std::convert::Into<{}>>(mut self, value: ArgType) -> Self {{",
            rust_name, attr_type
        )?;
    }
    writeln!(
        w,
        "        self.{} = ::std::option::Option::Some({});",
        rust_name, value
    )?;
    writeln!(w, "        self")?;
    writeln!(w, "    }}")?;
    Ok(())
}

fn write_build_fn<W: Write>(
    w: &mut W,
    op_name: &str,
    args: &[String],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<_> = args.iter().map(|arg| escaper.escape(arg)).collect();

    writeln!(w, "    /// Builds the `{}` operation.", op_name)?;
    write!(w, "    pub fn build<")?;
    for i in 0..args.len() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "O{}: ::std::convert::Into<crate::Output>", i)?;
    }
    write!(w, ">(&self, ")?;
    for (i, arg) in escaped_args.iter().enumerate() {
        write!(w, "{}: O{}, ", arg, i)?;
    }
    let scope_var = escaper.escape("scope");
    writeln!(
        w,
        r#"{scope}: &mut crate::Scope) -> crate::Result<crate::Operation> {{"#,
        scope = scope_var,
    )?;
    write!(w, "        self.build_impl(")?;
    for arg in &escaped_args {
        write!(w, "{}.into(), ", arg)?;
    }
    writeln!(w, "{})", scope_var)?;
    writeln!(w, "    }}")?;
    Ok(())
}

fn write_build_impl_fn<W: Write>(
    w: &mut W,
    op_name: &str,
    args: &[String],
    attrs: &[Attr],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<_> = args.iter().map(|arg| escaper.escape(arg)).collect();
    write!(w, "    fn build_impl(&self, ")?;
    for arg in &escaped_args {
        write!(w, "{}: crate::Output, ", arg)?;
    }
    let scope_var = escaper.escape("scope");
    let node_var = escaper.escape("nd");
    write!(
        w,
        r#"{scope}: &mut crate::Scope) -> crate::Result<crate::Operation> {{
        {scope}.new_operation({op_name:?}, |{node}| {{
"#,
        scope = scope_var,
        op_name = op_name,
        node = node_var,
    )?;
    for arg in escaped_args {
        writeln!(w, "            {}.add_input({});", node_var, arg)?;
    }
    writeln!(w, "            for op in &self.control_inputs {{")?;
    writeln!(w, "                {}.add_control_input(op);", node_var)?;
    writeln!(w, "            }}")?;
    for attr in attrs {
        write_set_attr(w, attr, &node_var)?;
    }
    writeln!(w, "            ::std::result::Result::Ok(())")?;
    writeln!(w, "        }})")?;
    writeln!(w, "    }}")?;
    Ok(())
}

fn write_attr<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    if attr.attr_type == "crate::Tensor" {
        writeln!(
            w,
            "    {}: ::std::option::Option<::std::boxed::Box<dyn crate::AnyTensor>>,",
            attr.rust_name
        )?;
    } else {
        writeln!(
            w,
            "    {}: ::std::option::Option<{}>,",
            attr.rust_name, attr.attr_type
        )?;
    }
    Ok(())
}

fn write_build_operation_struct<W: Write>(w: &mut W, op_name: &str) -> Result<(), io::Error> {
    writeln!(
        w,
        "/// An instance of '{}' Operation with it's Outputs and Inputs exposed as methods.",
        op_name
    )?;
    writeln!(w, "#[derive(Debug, Clone)]")?;
    writeln!(w, "pub struct {}Inst {{", op_name)?;
    writeln!(
        w,
        "    /// An instance of a fully built {} Operation in a Tensorflow graph.",
        op_name
    )?;
    writeln!(w, "    pub op: crate::Operation,")?;
    writeln!(w, "}}")?;

    Ok(())
}

///Takes a vector of dynamic_offset strings for inputs and outputs and returns a concatenated
///string of scalar_offsets to calculate the index of inputs and offsets by extracting the sum with
///the distributive property.
fn scalar_offsets(dynamic_offset: &[String], i: usize) -> String {
    let scalar_offsets = dynamic_offset
        .iter()
        .fold(HashMap::new(), |mut counts, string| {
            *counts.entry(string).or_insert(0) += 1;
            counts
        })
        .iter()
        .fold(String::new(), |mut scalar_offset, (string, count)| {
            //identity property
            if count > &1 {
                write!(scalar_offset, "{}*{}+", count, string).unwrap();
            } else {
                write!(scalar_offset, "{}+", string).unwrap();
            }
            scalar_offset
        });
    if scalar_offsets.is_empty() {
        format!("{}", i)
    } else {
        format!("{}{}", scalar_offsets, i)
    }
}

fn write_build_instance_fn<W: Write>(
    w: &mut W,
    op_name: &str,
    attrs: &[Attr],
    inputs: Vec<Input>,
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    writeln!(w, "    /// Builds a new instance of '{}' Operation with it's Outputs and Inputs exposed as methods.", op_name)?;
    write!(w, "    pub fn build_instance(&self, ")?;
    let mut seen_number_attr = HashMap::new();
    for input in &inputs {
        if input.number_attr.is_some() {
            write!(w, "{}: Vec<crate::Output>, ", input.clone().rust_name)?;
            seen_number_attr
                .entry(input.clone().number_attr.unwrap())
                .or_insert_with(|| input.clone());
        } else {
            write!(w, "{}: crate::Output, ", input.rust_name)?;
        }
    }
    let scope_var = escaper.escape("scope");
    write!(
        w,
        r#"{scope_var}: &mut crate::Scope) -> crate::Result<{op_name}Inst> {{
        let op = {scope_var}.new_operation({op_name:?}, |builder| {{
"#,
        op_name = op_name,
    )?;
    for input in &inputs {
        if input.number_attr.is_some() {
            //TODO: how are multiple lists handled here? may be an error with lower level protobuff
            //      bindings in OperationDescription. Is this ordered parameter wise internally?
            writeln!(
                w,
                "            builder.add_input_list(&{input});",
                input = input.rust_name,
            )?;
        } else {
            writeln!(w, "            builder.add_input({});", input.rust_name)?;
        }
    }

    for attr in attrs {
        if seen_number_attr.contains_key(&attr.rust_name) {
            writeln!(
                w,
                "            builder.set_attr_int(\"{}\", {}.clone().len() as i64)?;",
                attr.rust_name,
                seen_number_attr.get(&attr.rust_name).unwrap().rust_name
            )?;
        } else {
            write_set_attr(w, attr, "builder")?;
        }
    }
    writeln!(w, "            ::std::result::Result::Ok(())")?;
    writeln!(w, "        }})?;")?;
    writeln!(w, "        Ok({}Inst{{op}})", op_name)?;
    writeln!(w, "    }}")?;
    Ok(())
}

fn write_edge_method<T: Edge + Clone>(
    w: &mut impl Write,
    op_name: String,
    edge: T,
    i: usize,
    dynamic_offset: &mut Vec<String>,
) -> Result<(), io::Error> {
    let scalar_offsets = scalar_offsets(dynamic_offset, i);
    let edge_type = edge.edge_type();
    let rust_name = edge.rust_name();

    let mut op = "self.op.clone()";
    if edge_type == "Input" {
        op = "&self.op";
    }
    if let Some(number_attr) = &edge.number_attr() {
        //create a Vec<Edge> for this index
        writeln!(
            w,
            "    /// Returns a Vector of {} for '{}' {} of this {} operation.",
            rust_name, rust_name, edge_type, op_name
        )?;
        writeln!(
            w,
            "    pub fn {}(&self) -> crate::Result<Vec<crate::{}>>{{",
            rust_name, edge_type
        )?;
        if scalar_offsets.contains("self") {
            writeln!(
                w,
                "        let dynamic_offset = ({}) as i32;",
                scalar_offsets
            )?;
        }
        writeln!(w, "        let mut {}s = vec![];", edge_type)?;
        if dynamic_offset.is_empty() {
            writeln!(
                w,
                "        for i in {}..self.op.get_attr_int({:?})? as i32{{",
                i, number_attr
            )?;
            writeln!(
                w,
                "            {edge_type}s.push(crate::{edge_type} {{",
                edge_type = edge_type
            )?;
            writeln!(w, "                operation: {},", op)?;
            writeln!(w, "                index: i")?;
            writeln!(w, "            }});")?;
            writeln!(w, "        }}")?;
        } else {
            writeln!(
                w,
                "        for i in dynamic_offset..dynamic_offset+self.op.get_attr_int(\"{}\")? as i32{{",
                number_attr
            )?;
            writeln!(
                w,
                "            {edge_type}s.push(crate::{edge_type} {{",
                edge_type = edge_type
            )?;
            writeln!(w, "                operation: {},", op)?;
            writeln!(w, "                index: i")?;
            writeln!(w, "            }});")?;
            writeln!(w, "        }}")?;
        }
        writeln!(w, "        Ok({}s)", edge_type)?;
        writeln!(w, "    }}")?;
        //add the current self.op.get_attr_int(number_attr) to dynamic_offset to keep the current index into the Operations edges
        dynamic_offset.push(format!("self.op.get_attr_int(\"{}\")?", number_attr));
    } else {
        //create a single edge at the current dynamic_offset index
        writeln!(
            w,
            "    /// Returns the '{}' {} of this '{}' operation.",
            rust_name, edge_type, op_name
        )?;
        //if scalar_offsets is just the i value, we dont return a result since this is statically indexed
        if scalar_offsets == format!("{}", i) {
            writeln!(
                w,
                "    pub fn {}(&self) -> crate::{} {{",
                rust_name, edge_type
            )?;
            writeln!(w, "        crate::{} {{", edge_type)?;
            writeln!(w, "            operation: {},", op)?;
            writeln!(w, "            index: {}", i)?;
            writeln!(w, "        }}")?;
            writeln!(w, "    }}")?;
        } else {
            writeln!(
                w,
                "   pub fn {}(&self) -> crate::Result<crate::{}> {{",
                rust_name, edge_type
            )?;
            if scalar_offsets.contains("self") {
                writeln!(
                    w,
                    "        let dynamic_offset = ({}) as i32;",
                    scalar_offsets
                )?;
            }
            writeln!(w, "        Ok(crate::{} {{", edge_type)?;
            writeln!(w, "            operation: {},", op)?;
            writeln!(w, "            index: dynamic_offset")?;
            writeln!(w, "        }})")?;
            writeln!(w, "    }}")?;
        }
    }
    Ok(())
}

///writes the impl for the output struct that includes slicing implementations for Outputs that have output.number_attr set
fn write_build_instance_struct_impl<W: Write>(
    w: &mut W,
    op_name: &str,
    outputs: Vec<Output>,
    inputs: Vec<Input>,
) -> Result<(), io::Error> {
    writeln!(w, "impl {}Inst {{", op_name)?;

    let mut dynamic_offset: Vec<String> = vec![];
    //write methods for outputs
    for (i, output) in outputs.iter().enumerate() {
        write_edge_method(w, op_name.to_string(), output, i, &mut dynamic_offset)?;
    }
    //write methods for inputs
    let mut dynamic_offset: Vec<String> = vec![];
    for (i, input) in inputs.iter().enumerate() {
        write_edge_method(w, op_name.to_string(), input, i, &mut dynamic_offset)?;
    }

    writeln!(w, "}}")?;
    writeln!(w, "impl From<{op_name}Inst> for crate::Operation {{")?;
    writeln!(w, "    fn from(inst: {op_name}Inst) -> crate::Operation {{")?;
    writeln!(w, "        inst.op")?;
    writeln!(w, "    }}")?;
    writeln!(w, "}}")?;

    Ok(())
}

fn define_op<W: Write>(
    w: &mut W,
    keywords: &HashSet<String>,
    fn_escaper: &mut Escaper,
    struct_escaper: &mut Escaper,
    op: &OpDef,
) -> Result<(), io::Error> {
    let fn_name = fn_escaper.escape(&snake_name(&op.name));
    let name = struct_escaper.escape(&op.name);
    let op_name = op.name.clone();
    let mut op_outputs = vec![];
    let mut op_inputs = vec![];
    let args: Vec<_> = op.input_arg.iter().map(|arg| arg.name.clone()).collect();
    let mut attrs = Vec::new();
    let mut attr_escaper = Escaper::new(keywords);
    let mut output_escaper = Escaper::new(keywords);
    let mut input_escaper = Escaper::new(keywords);
    for attr in op.attr.iter() {
        let rust_type = match &attr.field_type as &str {
            // See OpDef.AttrDef.type in $TENSORFLOW/tensorflow/core/framework/op_def.proto
            // and AttrValue in $TENSORFLOW/tensorflow/core/framework/attr_value.proto
            "string" => "::std::string::String",
            "int" => "i64",
            "float" => "f32",
            "bool" => "bool",
            "type" => "crate::DataType",
            "shape" => "crate::Shape",
            "tensor" => "crate::Tensor",
            "func" => "::std::string::String",
            "list(string)" => "::std::vec::Vec<::std::string::String>",
            "list(int)" => "::std::vec::Vec<i64>",
            "list(float)" => "::std::vec::Vec<f32>",
            "list(bool)" => "::std::vec::Vec<bool>",
            "list(type)" => "::std::vec::Vec<crate::DataType>",
            "list(shape)" => "::std::vec::Vec<crate::Shape>",
            "list(tensor)" => "::std::vec::Vec<crate::Tensor>",
            "list(func)" => "::std::vec::Vec<::std::string::String>",
            t => {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "unrecognized field type {:?} for attribute {:?} of op {:?}",
                        t, &attr.name, &op_name
                    ),
                ))
            }
        };
        attrs.push(Attr {
            rust_name: attr_escaper.escape(&attr.name),
            attr_type: rust_type.to_string(),
            c_name: attr.name.clone(),
        });
    }

    for output in op.output_arg.iter() {
        let number_attr_opt = if output.number_attr.is_empty() {
            None
        } else {
            Some(output.number_attr.clone())
        };
        op_outputs.push(Output {
            rust_name: output_escaper.escape(&output.name),
            number_attr: number_attr_opt,
        });
    }
    for input in op.input_arg.iter() {
        let number_attr_opt = if input.number_attr.is_empty() {
            None
        } else {
            Some(input.number_attr.clone())
        };
        op_inputs.push(Input {
            rust_name: input_escaper.escape(&input.name),
            number_attr: number_attr_opt,
        });
    }

    writeln!(w, "/// Builder for the `{}` operation.", op_name)?;
    writeln!(w, "#[derive(::std::fmt::Debug, ::std::default::Default)]")?;
    writeln!(w, "pub struct {} {{", name)?;
    for attr in &attrs {
        write_attr(w, attr)?;
    }
    write!(
        w,
        r#"    control_inputs: ::std::vec::Vec<crate::Operation>,
}}
"#
    )?;
    write_build_operation_struct(w, &op_name)?;

    write!(
        w,
        r#"
impl {name} {{
    /// Creates a new `{name}`.
    pub fn new() -> Self {{
        Self::default()
    }}
"#,
        name = name
    )?;
    for attr in &attrs {
        write_attr_setter(w, attr)?;
    }
    write!(
        w,
        r#"
    /// Adds a control input.
    pub fn add_control_input(mut self, op: crate::Operation) -> Self {{
        self.control_inputs.push(op);
        self
    }}

"#
    )?;
    write_build_fn(w, &op_name, &args, keywords)?;
    write_build_impl_fn(w, &op_name, &args, &attrs, keywords)?;
    writeln!(w)?;
    write_build_instance_fn(w, &op_name, &attrs, op_inputs.clone(), keywords)?;
    writeln!(w, "}}")?;
    write_build_instance_struct_impl(w, &op_name, op_outputs, op_inputs)?;
    write_short_fn(w, &name, &fn_name, &args, keywords)?;
    writeln!(w)?;
    Ok(())
}

fn snake_name(name: &str) -> String {
    let mut s = String::new();
    let mut was_lower = false;
    for c in name.chars() {
        if c.is_uppercase() {
            if was_lower {
                s.push('_');
            }
            was_lower = false;
        } else {
            was_lower = true;
        }
        for cc in c.to_lowercase() {
            s.push(cc);
        }
    }
    s
}

struct Escaper<'a> {
    keywords: &'a HashSet<String>,
    used_names: HashSet<String>,
}

impl<'a> Escaper<'a> {
    fn new(keywords: &'a HashSet<String>) -> Self {
        Self {
            keywords,
            used_names: HashSet::new(),
        }
    }

    fn escape(&mut self, name: &str) -> String {
        let suffix = if self.keywords.contains(name) {
            "_"
        } else {
            ""
        };
        let mut candidate = format!("{}{}", name, suffix);
        let mut i = 2;
        while self.used_names.contains(&candidate) {
            candidate = format!("{}_{}", name, i);
            i += 1;
        }
        self.used_names.insert(candidate.clone());
        candidate
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let tensorflow_folder = Path::new(&args[1]);
    let output_folder = Path::new(&args[2]);
    let ops_bytes = fs::read(
        tensorflow_folder
            .join("tensorflow/core/ops/ops.pbtxt")
            .to_str()
            .ok_or("Unable to format path for tensorflow folder")?,
    )?;
    let ops = parser::parse(&ops_bytes).map_err(|e| {
        println!("Parse error at {:?}", e.pos);
        if let Some(p) = &e.pos {
            let input = String::from_utf8_lossy(&ops_bytes);
            println!("Previous: {}", &input[0..*p]);
            println!("Next: {}", &input[*p..]);
        }
        e
    })?;
    let keywords: HashSet<String> = [
        "abstract",
        "as",
        "async",
        "await",
        "become",
        "box",
        "break",
        "const",
        "continue",
        "crate",
        "do",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "final",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "macro",
        "match",
        "mod",
        "move",
        "mut",
        "override",
        "priv",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "try",
        "type",
        "typeof",
        "unsafe",
        "unsized",
        "use",
        "virtual",
        "where",
        "while",
        "yield",
        // These aren't technically keywords, but there doesn't appear to be a
        // way to refer to these types (e.g. qualified type names) if the name
        // has been shadowed by something else, so we treat them as keywords.
        "bool",
        "char",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "str",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        // build, new, and add_control_input aren't keywords, but they still
        // can't be used because they would clash with methods we're providing.
        "build",
        "new",
        "add_control_input",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut out = BufWriter::new(File::create(output_folder.join("src/ops/ops_impl.rs"))?);
    write!(
        &mut out,
        r#"// DO NOT EDIT. Generated by tensorflow-op-codegen/src/main.rs.
#![allow(
    non_snake_case,
    trivial_casts,
    unused_parens,
    unused_qualifications,
    unused_variables
)]

"#
    )?;
    let mut fn_escaper = Escaper::new(&keywords);
    let mut struct_escaper = Escaper::new(&keywords);
    for op in ops {
        define_op(
            &mut out,
            &keywords,
            &mut fn_escaper,
            &mut struct_escaper,
            &op,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_name() {
        assert_eq!(&snake_name("foo"), "foo");
        assert_eq!(&snake_name("fooBar"), "foo_bar");
        assert_eq!(&snake_name("FooBar"), "foo_bar");
        assert_eq!(&snake_name("abcXYZ"), "abc_xyz");
        assert_eq!(&snake_name("abcXYZdef"), "abc_xyzdef");
    }

    #[test]
    fn test_escaper() {
        let mut keywords = HashSet::new();
        keywords.insert("fn".to_string());
        let mut escaper = Escaper::new(&keywords);
        assert_eq!(&escaper.escape("fn"), "fn_");
        assert_eq!(&escaper.escape("fn"), "fn_2");
        assert_eq!(&escaper.escape("fn"), "fn_3");
        assert_eq!(&escaper.escape("foo"), "foo");
        assert_eq!(&escaper.escape("foo"), "foo_2");
    }
}
