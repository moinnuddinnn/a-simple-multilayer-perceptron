package main

import (
	"fmt"
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// Layer is a fully connected layer with a sigmoid activation.
type Layer struct {
	W   [][]float64 // weights, shape [out][in]
	B   []float64   // biases, shape [out]
	in  []float64   // cached input from the last forward pass
	out []float64   // cached activation from the last forward pass
}

func NewLayer(nIn, nOut int, rng *rand.Rand) *Layer {
	l := &Layer{W: make([][]float64, nOut), B: make([]float64, nOut)}
	scale := math.Sqrt(2.0 / float64(nIn+nOut)) // Xavier/Glorot init
	for i := range l.W {
		l.W[i] = make([]float64, nIn)
		for j := range l.W[i] {
			l.W[i][j] = rng.NormFloat64() * scale
		}
	}
	return l
}

func (l *Layer) Forward(x []float64) []float64 {
	l.in = x
	l.out = make([]float64, len(l.W))
	for i := range l.W {
		sum := l.B[i]
		for j, w := range l.W[i] {
			sum += w * x[j]
		}
		l.out[i] = sigmoid(sum)
	}
	return l.out
}

// Backward takes dLoss/dOutput, applies an SGD update, and returns dLoss/dInput.
func (l *Layer) Backward(gradOut []float64, lr float64) []float64 {
	gradIn := make([]float64, len(l.in))
	for i := range l.W {
		delta := gradOut[i] * l.out[i] * (1 - l.out[i]) // sigmoid'(z) = a(1-a)
		for j := range l.W[i] {
			gradIn[j] += delta * l.W[i][j] // use weight before the update
			l.W[i][j] -= lr * delta * l.in[j]
		}
		l.B[i] -= lr * delta
	}
	return gradIn
}

// MLP is a stack of layers.
type MLP struct{ Layers []*Layer }

// NewMLP builds a network from layer sizes, e.g. NewMLP(rng, 2, 4, 1).
func NewMLP(rng *rand.Rand, sizes ...int) *MLP {
	m := &MLP{}
	for i := 0; i < len(sizes)-1; i++ {
		m.Layers = append(m.Layers, NewLayer(sizes[i], sizes[i+1], rng))
	}
	return m
}

func (m *MLP) Predict(x []float64) []float64 {
	for _, l := range m.Layers {
		x = l.Forward(x)
	}
	return x
}

// TrainStep runs one forward/backward pass on a single sample and returns its MSE loss.
func (m *MLP) TrainStep(x, target []float64, lr float64) float64 {
	out := m.Predict(x)

	loss := 0.0
	grad := make([]float64, len(out))
	for i := range out {
		diff := out[i] - target[i]
		loss += diff * diff
		grad[i] = 2 * diff / float64(len(out))
	}
	loss /= float64(len(out))

	for i := len(m.Layers) - 1; i >= 0; i-- {
		grad = m.Layers[i].Backward(grad, lr)
	}
	return loss
}

func main() {
	rng := rand.New(rand.NewSource(42))

	// XOR dataset
	X := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	Y := [][]float64{{0}, {1}, {1}, {0}}

	net := NewMLP(rng, 2, 4, 1) // 2 inputs -> 4 hidden -> 1 output

	const epochs = 10000
	const lr = 1.0

	for epoch := 1; epoch <= epochs; epoch++ {
		total := 0.0
		for _, i := range rng.Perm(len(X)) { // shuffle each epoch
			total += net.TrainStep(X[i], Y[i], lr)
		}
		if epoch%1000 == 0 {
			fmt.Printf("epoch %5d  loss %.6f\n", epoch, total/float64(len(X)))
		}
	}

	fmt.Println("\nPredictions:")
	for i, x := range X {
		p := net.Predict(x)[0]
		fmt.Printf("  %v -> %.4f (target %v)\n", x, p, Y[i][0])
	}
}
