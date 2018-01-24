library(shiny)
library(shinyjs)
library(plotly)

shinyUI(
    fluidPage(
        useShinyjs(),
        fluidRow(
            column(
                width = 6,offset=2,
                h3("Scatterplot"),
                HTML("<h5><i>Created using ggplot2, hrbrthemes and ggsci on the iris data.</i></h5>"),
                plotOutput("scatterplot", width=800)
            )
        )

    )
)
