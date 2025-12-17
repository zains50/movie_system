import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Outlet } from "react-router-dom";
import { useParams } from "react-router-dom";
import { useState } from "react"
import MovieData from "./../assets/movie_info.json"
import TopBar from "./../TopBar/TopBar"

import "./MoviePage.css"

export function MovieDefaultPage(){
    return (
        <div className="MovieDefaultPage"  >
            <TopBar/>
            <Outlet/>
        </div>
    )
}

async function loadWatchlist() {
  const res = await fetch("/watch_list.txt");
  const text = await res.text();
  return text.split(",").map(Number); // convert to numbers
}



export default function MoviePage() {
  const { movie_id } = useParams();
  const poster_link = `/posters/${movie_id}.jpg`
  const movie_list_text = "/watch_list.txt" 

  const movie_json = MovieData[movie_id]
  const movie_name = movie_json["Title"]
  const year = movie_json["Year"]
  const plot = movie_json["Plot"]
  const genres = movie_json["Genre"]
  const imdbRating = movie_json["imdbRating"]
  const numRatings = movie_json["imdbVotes"]
  const boxOffice = movie_json["BoxOffice"]
  const actors = movie_json["Actors"]

  const [inWatchList, setInWatchList] = useState(false)

  

  function handleClick(){
    setInWatchList(prev => !prev)
    

    if (setInWatchList) {
      text_watch_list = loadWatchlist()
      console.log(text_watch_list)
    }

  }


  return (
    <div className="MoviePage">
      <div className="MoviePosterImageContainer">
      <img src={poster_link} alt={poster_link} />
      <button onClick={handleClick} className={inWatchList ? "button_one" : "button_zero"}> {inWatchList ? "Remove from watch list" : "Add to watchlist"} </button>
      </div>

      <div className="MovieInfoContainer">
        <header>Movie: {movie_name} ({year})</header>
        <section>Plot: {plot}</section>
        <section>Genres : {genres}</section>
        <section>IMDB Rating : {imdbRating}</section>
        <section>Ratings: {numRatings}</section>
        <section>Box Office: {boxOffice}</section>
        <section>Actors : {actors}</section>
      </div>
    </div>
  );
}