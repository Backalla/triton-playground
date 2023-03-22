
ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

lazy val root = (project in file("."))
  .settings(
    name := "triton-scala",
    idePackagePrefix := Some("org.booking"),
    libraryDependencies ++= Seq(
      "org.bytedeco" % "tritonserver-platform" % "2.26-1.5.8",
      "org.bytedeco" % "javacpp" % "1.5.8"
    ),
    assembly / assemblyMergeStrategy := {
      case x if Assembly.isConfigFile(x) =>
        MergeStrategy.concat

      case PathList(ps @ _*) if Assembly.isReadme(ps.last) || Assembly.isLicenseFile(ps.last) =>
        MergeStrategy.rename

      case PathList("META-INF", xs @ _*) =>
        xs map {_.toLowerCase} match {
          case "manifest.mf" :: Nil | "index.list" :: Nil | "dependencies" :: Nil =>
            MergeStrategy.discard
          case "services" :: _ =>
            MergeStrategy.filterDistinctLines
          case "io.netty.versions.properties" :: Nil =>
            MergeStrategy.first
          case ps @ _ :: _ if ps.last.equals("pom.xml") || ps.last.equals("pom.properties") =>
            MergeStrategy.first
          case ps @ _ :: _ if ps.last.equals("module-info.class") =>
            MergeStrategy.discard
          case ps @ _ :: _ if ps.last.equals("log4j2plugins.dat") =>
            MergeStrategy.first
          case ps @ _ :: _ if ps.last.equals("jni-config.json") =>
            MergeStrategy.first
          case ps @ _ :: _ if ps.last.equals("reflect-config.json") =>
            MergeStrategy.first
          case _ => MergeStrategy.deduplicate
        }

      case _ =>
        MergeStrategy.deduplicate
    }
  )
