"use client";
import Header from "./components/header";
import DashboardBody from "./components/dashboardBody";
import { withPageAuthRequired } from "@auth0/nextjs-auth0";

export default withPageAuthRequired(function Home() {

  return (
    <>
      <Header />
      <DashboardBody />
    </>
  );
})
