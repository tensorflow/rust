(function() {var implementors = {
"half":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for &amp;<a class=\"struct\" href=\"half/struct.f16.html\" title=\"struct half::f16\">f16</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for <a class=\"struct\" href=\"half/struct.bf16.html\" title=\"struct half::bf16\">bf16</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for <a class=\"struct\" href=\"half/struct.f16.html\" title=\"struct half::f16\">f16</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for &amp;<a class=\"struct\" href=\"half/struct.bf16.html\" title=\"struct half::bf16\">bf16</a>"]],
"ndarray":[["impl&lt;'a, A, S, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for &amp;'a <a class=\"struct\" href=\"ndarray/struct.ArrayBase.html\" title=\"struct ndarray::ArrayBase\">ArrayBase</a>&lt;S, D&gt;<span class=\"where fmt-newline\">where\n    <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.71.1/std/primitive.reference.html\">&amp;'a A</a>: 'a + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a>&lt;Output = A&gt;,\n    S: <a class=\"trait\" href=\"ndarray/trait.Data.html\" title=\"trait ndarray::Data\">Data</a>&lt;Elem = A&gt;,\n    D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,</span>"],["impl&lt;A, S, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for <a class=\"struct\" href=\"ndarray/struct.ArrayBase.html\" title=\"struct ndarray::ArrayBase\">ArrayBase</a>&lt;S, D&gt;<span class=\"where fmt-newline\">where\n    A: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a>&lt;Output = A&gt;,\n    S: <a class=\"trait\" href=\"ndarray/trait.DataOwned.html\" title=\"trait ndarray::DataOwned\">DataOwned</a>&lt;Elem = A&gt; + <a class=\"trait\" href=\"ndarray/trait.DataMut.html\" title=\"trait ndarray::DataMut\">DataMut</a>,\n    D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,</span>"]],
"num_complex":[["impl&lt;'a, T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + <a class=\"trait\" href=\"num_traits/trait.Num.html\" title=\"trait num_traits::Num\">Num</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a>&lt;Output = T&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for &amp;'a <a class=\"struct\" href=\"num_complex/struct.Complex.html\" title=\"struct num_complex::Complex\">Complex</a>&lt;T&gt;"],["impl&lt;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> + <a class=\"trait\" href=\"num_traits/trait.Num.html\" title=\"trait num_traits::Num\">Num</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a>&lt;Output = T&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for <a class=\"struct\" href=\"num_complex/struct.Complex.html\" title=\"struct num_complex::Complex\">Complex</a>&lt;T&gt;"]],
"tensorflow":[["impl&lt;T: <a class=\"trait\" href=\"tensorflow/trait.TensorType.html\" title=\"trait tensorflow::TensorType\">TensorType</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.71.1/core/ops/arith/trait.Neg.html\" title=\"trait core::ops::arith::Neg\">Neg</a> for <a class=\"struct\" href=\"tensorflow/expr/struct.Expr.html\" title=\"struct tensorflow::expr::Expr\">Expr</a>&lt;T&gt;"]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()