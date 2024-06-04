

all: build

build:
	go build -o gonn && ls ./gonn

run:
	@go run main.go

clean:
	rm -f ./gonn

# Test the application
test:
	@echo "Testing..."
	@go test ./tests -v


