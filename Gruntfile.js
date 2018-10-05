module.exports = function (grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON("package.json"),
        exec: {
          build: {
            cmd: "Rscript -e \"bookdown::render_book('index.Rmd', 'bookdown::gitbook')\""
          }
        }
    });
    grunt.loadNpmTasks("grunt-exec");
    grunt.registerTask("default", ["exec"]);
};
