TEMPLATE_RUN="run-initial.sh"
TEMPLATE_BUILD="build-initial.sh"

# Copy the template run script to the src/
cp "$TEMPLATE_RUN" "src/run.sh"

# Copy the template build script to the src/
cp "$TEMPLATE_BUILD" "src/build.sh"

# Build the project
./devtool build_project

# Submit the build
python3 telerun.py submit build.tar