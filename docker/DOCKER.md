## Production images

TODO

For Blackwell image build, see [Build_instructions_blackwell.md](Build_instructions_blackwell.md)

## Development images

These images are the biggest but come with all the build tooling, needed to compile things at runtime (Deepspeed)

```
docker build \
    -f docker/Dockerfile \
    --target devel \
    -t openfold-docker:devel .
```

## Test images

Build the test image
```
docker build \
    -f docker/development/Dockerfile \
    --target test \
    -t openfold-docker:test .
```

Run the unit tests
```
docker run \
    --rm \
    -v $(pwd -P):/opt/openfold3 \
    -t openfold-docker:test \
    pytest openfold3/tests -vvv
```
