package gonn

type NeuralNetworkNode struct {
	InputNode  *NeuralNetworkNode
	OutputNode *NeuralNetworkNode
}

type Forwardable interface {
	forward() float64
}

func (nnn *NeuralNetworkNode) forward() {

}
