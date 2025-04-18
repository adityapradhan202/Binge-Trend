  const express = require("express");
  const path = require("path");
  const axios = require("axios");

  const app = express();
  const port = 8080;

  app.set("view engine", "ejs");
  app.set("views", path.join(__dirname, "views"));
  app.use(express.static("public"));
  app.use(express.urlencoded({ extended: true }));
  app.use(express.json());
  app.use(express.static('assets'));


  app.get("/", (req, res) => {
    res.render("index", { prediction: "" });
  });

  app.get("/Synopsis_to_Movie", (req, res) => {
    res.render("Synopsis_to_movie", { movies: [] }); 
  });
  

  app.post("/Synopsis_to_Movie", async (req, res) => {
    const { synopsis } = req.body;

    try {
      const response = await axios.post("http://127.0.0.1:6000/find-match", { synopsis });
      const movies = Object.values(response.data); 
      res.render("Synopsis_to_movie", { movies });
    } catch (error) {
      console.error("Error fetching from Flask API:", error.message);
      res.render("Synopsis_to_movie", { movies: [] });
    }
  });


  app.get("/Synopsis_to_Genre", (req, res) => {
    res.render("Synopsis_to_genre", { prediction: "" });
  });


  app.post("/submit", async (req, res) => {
    const { synopsis } = req.body;

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", { synopsis });
      res.json({ message: response.data });
    } catch (err) {
      console.error(err.message);
      res.json({ message: "Something went wrong. Please try again." });
    }
  });

  app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
  });
