use super::Filter;

#[derive(Clone, Debug)]
pub struct AlphaBetaFilter {
    alpha: f32,
    beta: f32,
    x: f32,
    /// Predicted change in `x` per call to `push`.
    v: f32,
}

impl AlphaBetaFilter {
    pub fn new(alpha: f32, beta: f32) -> Self {
        assert!(0.0 < alpha && alpha < 1.0);
        assert!(0.0 < beta && beta < 1.0);
        Self {
            alpha,
            beta,
            x: 0.0,
            v: 0.0,
        }
    }
}

impl Filter<f32> for AlphaBetaFilter {
    fn push(&mut self, value: f32) -> f32 {
        let prediction = self.x + self.v;
        let residual = value - prediction;

        self.x = prediction + self.alpha * residual;
        self.v = self.v + self.beta * residual;

        self.x
    }

    fn reset(&mut self) {
        self.x = 0.0;
        self.v = 0.0;
    }
}
