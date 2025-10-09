import Image from "next/image";
import Link from "next/link"
import Header from "./components/header";
import LandingBody from "./components/landingBody";
import Features from "./components/features";


export default function Home() {
  return (
    <>
      <Header />
      <LandingBody />
      <Features />
    </>
  );
}
