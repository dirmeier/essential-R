module.exports = function (grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON("package.json"),
        exec: {
          build: {
            cmd: "Rscript -e 'rmarkdown::render_site()'"
          }
        }
    });
    grunt.loadNpmTasks("grunt-exec");
    grunt.registerTask("default", ["exec"]);
};
