import './App.css'
import { Routes, Route, BrowserRouter } from 'react-router-dom'

import HomePage from "./HomePage/HomePage"
import MoviePage from './MoviePage/MoviePage'
import WatchList from "./WatchList/WatchList"
import { MovieDefaultPage } from "./MoviePage/MoviePage"


export default function App() {



  return (
    <div className="app">
    <Routes>
      <Route path="/" element={<HomePage/>}/>
      <Route path="/home" element={<HomePage/>}/>
      <Route path="/movie" element={<MovieDefaultPage/>}>
          <Route path=":movie_id" element={<MoviePage />} />
      </Route>
      <Route path="/watch_list" element = {<WatchList/>}/>
    </Routes>
    </div>

 ) 
 
}

