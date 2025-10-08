import Image from "next/image";
import Link from "next/link"

export default function Home() {
  return (
    <>
    <h1>FrontPage TODO (jacob note will grind this out by friday)</h1>
    <div className="routes">
      <Link href="/canvas">
        <button>Canvas Entry</button>
      </Link>
      <Link href="/login">
        <button>Login page</button>
      </Link>
    </div>
    </>
  );
}
