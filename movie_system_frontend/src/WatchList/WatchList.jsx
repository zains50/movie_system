import MoviePlate from "./../MoviePlate/MoviePlate";
import TopBar from "./../TopBar/TopBar";
import { useEffect, useState } from "react";

async function loadWatchlist() {
  const res = await fetch("/watch_list.txt");
  const text = await res.text();
  return text.split(",").map(Number); // convert to numbers
}

export default function WatchList() {
  const [movieList, setMovieList] = useState([]);

  useEffect(() => {
    loadWatchlist().then((list) => {
      console.log("Loaded watchlist:", list);
      setMovieList(list);
    });
  }, []); // run ONCE on mount

  return (
    <>
      <TopBar />
      <div className="MoviesTab">
        {movieList.map((movie) => (
          <MoviePlate key={movie} movie_id={movie} />
        ))}
      </div>
    </>
  );
}
