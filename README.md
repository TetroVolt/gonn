
# GONN

## Go Neural Networks

## Description
Go Neural Networks is a demo of simple neural networks from scratch in GO.

My primary motivation in writing this was to learn GO (I am still a novice in GO).

My goal was **NOT** to not build a performant production ready GO ML framework.

This was written just for fun and to sanity check my understanding of
basic Neural Networks, backpropagation, and optimizer algorithms. 

And to learn the basics about Go error handling, packages, and unit tests.

## Milestones

- [x] Basic Matrix library 
- [x] Basic Perceptron XOR demo
- [ ] (Not Started) Abtract graph NN module

## Running
This is a work in progress. So far only a small XOR perceptron demo exists called from `main.go`
This project uses a [Makefile](https://www.gnu.org/software/make/manual/make.html)

### Running main
```sh
make run
```
or
```sh
go run main.go
```

### Building
```sh
make build
```

### Unit Tests
```sh
make test
```
or 
```sh
go test ./tests/ -v
```


