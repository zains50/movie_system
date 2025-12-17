import "./TopBar.css"


export default function Navbar() {

  
  return (
    // acts as the main container for the bar, provides navigation links
    <nav className="navbar">
     
      <div className="logo">Movie Recommender</div>
        <div className="search-box"> 
            <input
                type="search"
                id="search-form"
                className="search-input" 
                placeholder="Search items..."
            />
        </div>

      <ul className="nav-links">
        <li><a href="/">Home</a></li>
        <li><a href="/Watch_List">Watch List</a></li>
        <li><a href="/Sign_In">Sign Up</a></li>
        <li><a href="/contact">Contact</a></li>
      </ul>
    </nav>
  );
}
