#![recursion_limit = "128"]
//! The package provides macros for internal usage in TensorFlow. No backwards
//! compatibility guarantees are made.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::Literal;
use proc_macro2::Span;
use quote::quote;
use quote::ToTokens;
use syn::braced;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::Error;
use syn::Ident;
use syn::LitStr;
use syn::Result;
use syn::Token;
use syn::Type;

#[derive(Clone)]
struct Arg {
    name: Ident,
}

impl Parse for Arg {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse()?;
        Ok(Arg { name })
    }
}

struct Args {
    args: Punctuated<Arg, Token![,]>,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> Result<Self> {
        let list;
        braced!(list in input);
        Ok(Args {
            args: list.parse_terminated(Arg::parse)?,
        })
    }
}

#[derive(Clone)]
struct Attr {
    optional: bool,
    rust_name: Ident,
    attr_type: Type,
    c_name: LitStr,
}

impl Parse for Attr {
    fn parse(input: ParseStream) -> Result<Self> {
        let rust_name = input.parse()?;
        let mut optional = false;
        let lookahead = input.lookahead1();
        if lookahead.peek(Token![?]) {
            input.parse::<Token![?]>()?;
            optional = true;
        }
        input.parse::<Token![:]>()?;
        let attr_type = input.parse()?;
        input.parse::<Token![=>]>()?;
        let c_name = input.parse()?;
        Ok(Attr {
            optional,
            rust_name,
            attr_type,
            c_name,
        })
    }
}

struct Attrs {
    attrs: Punctuated<Attr, Token![,]>,
}

impl Parse for Attrs {
    fn parse(input: ParseStream) -> Result<Self> {
        let list;
        braced!(list in input);
        Ok(Attrs {
            attrs: list.parse_terminated(Attr::parse)?,
        })
    }
}

struct DefineOpInput {
    fn_name: Ident,
    name: Ident,
    op_name: LitStr,
    args: Vec<Arg>,
    attrs: Vec<Attr>,
}

impl Parse for DefineOpInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let fn_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let name = input.parse()?;
        input.parse::<Token![,]>()?;
        let op_name = input.parse()?;
        let mut args = Vec::new();
        let mut attrs = Vec::new();
        loop {
            let lookahead = input.lookahead1();
            if !lookahead.peek(Token![,]) {
                break;
            }
            input.parse::<Token![,]>()?;
            let ident: Ident = input.parse()?;
            if ident == "args" {
                let new_args: Args = input.parse()?;
                args.extend(new_args.args);
            } else if ident == "attrs" {
                let new_attrs: Attrs = input.parse()?;
                attrs.extend(new_attrs.attrs);
            } else {
                return Err(Error::new(Span::call_site(), "expected `attrs` or `args`"));
            }
        }
        Ok(DefineOpInput {
            fn_name,
            name,
            op_name,
            args,
            attrs,
        })
    }
}

struct AttrDefs<'a>(&'a [Attr]);

impl<'a> ToTokens for AttrDefs<'a> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for attr in self.0 {
            let rust_name = &attr.rust_name;
            let attr_type = &attr.attr_type;
            if attr.optional {
                tokens.extend(quote! { #rust_name: ::std::option::Option<#attr_type>, });
            } else {
                tokens.extend(quote! { #rust_name: #attr_type, });
            }
        }
    }
}

struct AttrSetters<'a>(&'a [Attr]);

impl<'a> ToTokens for AttrSetters<'a> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for attr in self.0 {
            let comment =
                Literal::string(&format!("Sets the `{}` attribute.", attr.c_name.value()));
            let rust_name = &attr.rust_name;
            let attr_type = &attr.attr_type;
            let mut needs_into = false;
            let mut arg_type = attr_type.clone();
            if attr_type == &syn::parse_str::<Type>("String").unwrap() {
                needs_into = true;
                // TODO: don't use parse
                arg_type = syn::parse_str::<Type>("&str").unwrap()
            };
            let mut value = quote! { value };
            if needs_into {
                value = quote! { <#arg_type as ::std::convert::Into<#attr_type>>::into(#value) };
            }
            if attr.optional {
                value = quote! { ::std::option::Option::Some(#value) };
            }
            tokens.extend(quote! {
                #[doc = #comment]
                pub fn #rust_name(mut self, value: #arg_type) -> Self {
                    self.#rust_name = #value;
                    self
                }
            });
        }
    }
}

struct BuildFnGenerics {
    arg_count: usize,
}

impl ToTokens for BuildFnGenerics {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        if self.arg_count == 0 {
            return;
        }
        tokens.extend(quote! {<});
        for i in 0..self.arg_count {
            if i > 0 {
                tokens.extend(quote! {,});
            }
            let arg = Ident::new(&format!("O{}", i + 1), Span::call_site());
            tokens.extend(quote! {#arg: ::std::convert::Into<crate::Output>});
        }
        tokens.extend(quote! {>});
    }
}

struct BuildFnArgs<'a> {
    args: &'a [Arg],
}

impl<'a> ToTokens for BuildFnArgs<'a> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for (i, arg) in self.args.iter().enumerate() {
            let arg_name = &arg.name;
            let arg_type = Ident::new(&format!("O{}", i + 1), Span::call_site());
            tokens.extend(quote! {#arg_name: #arg_type, });
        }
    }
}

struct SetAttr<'a> {
    attr: &'a Attr,
}

impl<'a> ToTokens for SetAttr<'a> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let c_name = &self.attr.c_name;
        let rust_name = &self.attr.rust_name;
        let setter = |value| match self
            .attr
            .attr_type
            .clone()
            .into_token_stream()
            .to_string()
            .as_str()
        {
            "String" => quote! { nd.set_attr_string(#c_name, &#value)?; },
            "DataType" => quote! { nd.set_attr_type(#c_name, #value)?; },
            "bool" => quote! { nd.set_attr_bool(#c_name, #value)?; },
            "i64" => quote! { nd.set_attr_int(#c_name, #value)?; },
            "Shape" => quote! { nd.set_attr_shape(#c_name, &#value)?; },
            ty => panic!(
                "Unrecognized attribute type for {}: {}",
                self.attr.rust_name, ty
            ),
        };
        tokens.extend(if self.attr.optional {
            let set = setter(quote! { *value });
            quote! {
                if let Some(value) = &self.#rust_name {
                    #set
                }
            }
        } else {
            setter(quote! { self.#rust_name })
        });
    }
}

struct BuildFn<'a> {
    op_name: &'a LitStr,
    args: &'a [Arg],
    attrs: &'a [Attr],
}

impl<'a> ToTokens for BuildFn<'a> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let op_name = &self.op_name;
        let build_fn_generics = BuildFnGenerics {
            arg_count: self.args.len(),
        };
        let build_fn_args = BuildFnArgs { args: &self.args };
        let arg_names = self.args.iter().map(|arg| &arg.name);
        let set_attrs = self.attrs.iter().map(|attr| SetAttr { attr });
        tokens.extend(quote! {
            #[doc = "Builds the `"]
            #[doc = #op_name]
            #[doc = "` operation."]
            pub fn build#build_fn_generics(&self, #build_fn_args scope: &mut crate::Scope) -> crate::Result<crate::Operation> {
                let name = scope.get_unique_name_for_op(#op_name);
                let mut graph = scope.graph_mut();
                let mut nd = graph.new_operation(#op_name, &name)?;
                #(
                    nd.add_input(#arg_names);
                )*
                for op in &self.control_inputs {
                    nd.add_control_input(op);
                }
                #(#set_attrs)*
                nd.finish()
            }
        });
    }
}

struct ShortFn<'a> {
    name: &'a Ident,
    fn_name: &'a Ident,
    args: &'a [Arg],
}

impl<'a> ToTokens for ShortFn<'a> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let name = &self.name;
        let fn_name = &self.fn_name;
        let build_fn_generics = BuildFnGenerics {
            arg_count: self.args.len(),
        };
        let build_fn_args = BuildFnArgs { args: &self.args };
        let arg_names = self.args.iter().map(|arg| &arg.name);
        let mut docs = format!("Shorthand for `{}::new().build(scope)", name);
        for arg in self.args {
            docs.push_str(", ");
            docs.push_str(&arg.name.to_string());
        }
        docs.push_str(")`.");
        tokens.extend(quote! {
            #[doc = #docs]
            pub fn #fn_name#build_fn_generics(#build_fn_args scope: &mut crate::Scope) -> crate::Result<crate::Operation> {
                #name::new().build(#(#arg_names, )* scope)
            }
        });
    }
}

#[proc_macro]
pub fn define_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DefineOpInput);
    let fn_name = input.fn_name;
    let name = input.name;
    let op_name = input.op_name;
    let name_str = name.to_string();
    let name_str_plus_period = name_str + ".";
    let attr_defs = AttrDefs(&input.attrs);
    let attr_setters = AttrSetters(&input.attrs);
    let build_fn = BuildFn {
        op_name: &op_name,
        args: &input.args,
        attrs: &input.attrs,
    };
    let short_fn = ShortFn {
        name: &name,
        fn_name: &fn_name,
        args: &input.args,
    };
    let stream = quote! {
        #[doc = "Builder for the `"]
        #[doc = #op_name]
        #[doc = "` operation."]
        #[derive(Debug,Default)]
        pub struct #name {
            #attr_defs
            control_inputs: Vec<crate::Operation>,
        }

        impl #name {
            #[doc = "Creates a new"]
            #[doc = #name_str_plus_period]
            pub fn new() -> Self {
                Self::default()
            }

            #attr_setters

            /// Adds a control input.
            pub fn add_control_input(mut self, op: crate::Operation) -> Self {
                self.control_inputs.push(op);
                self
            }

            #build_fn
        }

        #short_fn
    };
    stream.into()
}
