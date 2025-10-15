use tract_onnx::prelude::*;
use ndarray::Array4;

pub struct TractRunner {
    model: TypedRunnableModel<TypedModel>,
}

impl TractRunner {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact(&[1, 3, 256, 256]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    pub fn run(&self, input: Array4<f32>) -> anyhow::Result<Array4<f32>> {
        let input: Tensor = tract_ndarray::Array4::from(input).into();
        let result = self.model.run(tvec!(input))?;
        let array: tract_ndarray::Array4<f32> = result[0].to_array_view::<f32>()?.to_owned().into();
        Ok(array.into())
    }
}
