use std::collections::HashSet;
use std::env;
use std::error::Error;
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
    write!(
        w,
        "        if let ::std::option::Option::Some(value) = &self.{} {{\n",
        rust_name
    )?;
    write!(w, "            {};\n", setter)?;
    write!(w, "        }}\n")?;
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
    let escaped_args: Vec<_> = args.iter().map(|arg| escaper.escape(&arg)).collect();
    write!(w, "/// Shorthand for `{}::new().build(", name)?;
    for arg in &escaped_args {
        write!(w, "{}, ", &arg)?;
    }
    let scope_var = escaper.escape("scope");
    write!(w, "{})`.\n", scope_var)?;
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
    write!(
        w,
        "{}: &mut crate::Scope) -> crate::Result<crate::Operation> {{\n",
        scope_var
    )?;
    write!(w, "    {}::new().build(", name)?;
    for arg in escaped_args {
        write!(w, "{}, ", arg)?;
    }
    write!(w, "{})\n", scope_var)?;
    write!(w, "}}\n")?;
    Ok(())
}

fn write_attr_setter<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    write!(w, "\n")?;
    write!(w, "    /// Sets the `{}` attribute.\n", &attr.c_name)?;
    let rust_name = &attr.rust_name;
    let attr_type = &attr.attr_type;
    let mut value = "value.into()".to_string();
    if attr_type == "crate::Tensor" {
        value = format!(
            "(::std::boxed::Box::new({}) as ::std::boxed::Box<dyn crate::AnyTensor>)",
            value
        );
        write!(
            w,
            "    pub fn {}<T: crate::TensorType, ArgType: ::std::convert::Into<crate::Tensor<T>>>(mut self, value: ArgType) -> Self {{\n",
            rust_name
        )?;
    } else {
        write!(
            w,
            "    pub fn {}<ArgType: ::std::convert::Into<{}>>(mut self, value: ArgType) -> Self {{\n",
            rust_name, attr_type
        )?;
    }
    write!(
        w,
        "        self.{} = ::std::option::Option::Some({});\n",
        rust_name, value
    )?;
    write!(w, "        self\n")?;
    write!(w, "    }}\n")?;
    Ok(())
}

fn write_build_fn<W: Write>(
    w: &mut W,
    op_name: &str,
    args: &[String],
    keywords: &HashSet<String>,
) -> Result<(), io::Error> {
    let mut escaper = Escaper::new(keywords);
    let escaped_args: Vec<_> = args.iter().map(|arg| escaper.escape(&arg)).collect();

    write!(w, "    /// Builds the `{}` operation.\n", op_name)?;
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
    write!(
        w,
        r#"{scope}: &mut crate::Scope) -> crate::Result<crate::Operation> {{
"#,
        scope = scope_var,
    )?;
    write!(w, "        self.build_impl(")?;
    for arg in &escaped_args {
        write!(w, "{}.into(), ", arg)?;
    }
    write!(w, "{})\n", scope_var)?;
    write!(w, "    }}\n")?;
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
    let escaped_args: Vec<_> = args.iter().map(|arg| escaper.escape(&arg)).collect();
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
        write!(w, "            {}.add_input({});\n", node_var, arg)?;
    }
    write!(w, "            for op in &self.control_inputs {{\n")?;
    write!(w, "                {}.add_control_input(op);\n", node_var)?;
    write!(w, "            }}\n")?;
    for attr in attrs {
        write_set_attr(w, attr, &node_var)?;
    }
    write!(w, "            ::std::result::Result::Ok(())\n")?;
    write!(w, "        }})\n")?;
    write!(w, "    }}\n")?;
    Ok(())
}

fn write_attr<W: Write>(w: &mut W, attr: &Attr) -> Result<(), io::Error> {
    if attr.attr_type == "crate::Tensor" {
        write!(
            w,
            "    {}: ::std::option::Option<::std::boxed::Box<dyn crate::AnyTensor>>,\n",
            attr.rust_name
        )?;
    } else {
        write!(
            w,
            "    {}: ::std::option::Option<{}>,\n",
            attr.rust_name, attr.attr_type
        )?;
    }
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
    let args: Vec<_> = op.input_arg.iter().map(|arg| arg.name.clone()).collect();
    let mut attrs = Vec::new();
    let mut attr_escaper = Escaper::new(keywords);
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
    write!(w, "/// Builder for the `{}` operation.\n", op_name)?;
    write!(w, "#[derive(::std::fmt::Debug, ::std::default::Default)]\n")?;
    write!(w, "pub struct {} {{\n", name)?;
    for attr in &attrs {
        write_attr(w, &attr)?;
    }
    write!(
        w,
        r#"    control_inputs: ::std::vec::Vec<crate::Operation>,
}}

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
    write_build_fn(w, &op_name, &args, &keywords)?;
    write!(w, "\n")?;
    write_build_impl_fn(w, &op_name, &args, &attrs, &keywords)?;
    write!(w, "}}\n")?;
    write!(w, "\n")?;
    write_short_fn(w, &name, &fn_name, &args, &keywords)?;
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
        "{}",
        r#"// DO NOT EDIT. Generated by tensorflow-op-codegen/src/main.rs.
#![allow(
    non_snake_case,
    trivial_casts,
    unused_parens,
    unused_qualifications
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
