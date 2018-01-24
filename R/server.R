library(shiny)
library(shinyjs)
library(ggplot2)
library(ggsci)
library(hrbrthemes)

shinyServer(function(input, output)
{

    output$scatterplot <- renderPlot({
        ggplot(iris, aes(x=Sepal.Width, y=Sepal.Length, colour=Species)) +
            geom_point(aes(colour=Species)) +
            geom_smooth(alpha=.1) +
            hrbrthemes::theme_ipsum_rc() +
            ggsci::scale_colour_rickandmorty() +
            labs(title="Iris scatter plot",
                 subtitle="Visualizing sepal width vs. sepal length for several species in a schwifty theme.")
  })
})
